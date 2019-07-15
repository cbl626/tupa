from collections import deque

from graph import Graph

from .edge import StateEdge
from .node import StateNode
from ..action import Actions
from ..config import Config, requires_node_labels, requires_node_properties, requires_edge_attributes, requires_anchors
from ..constraints.validation import CONSTRAINTS, Constraints, Direction, ROOT_ID, ROOT_LAB, ANCHOR_LAB, DEFAULT_LABEL
from ..model import NODE_LABEL_KEY, NODE_PROPERTY_KEY, EDGE_ATTRIBUTE_KEY


class InvalidActionError(AssertionError):
    def __init__(self, *args, is_type=False):
        super().__init__(*args)
        self.is_type = is_type


class State:
    """
    The parser's state, responsible for applying actions and creating the final Graph
    """
    def __init__(self, graph=None, conllu=None, target=None):
        """
        :param graph: Graph for training, not needed for testing except for getting the graph id
        :param conllu: Graph with node per token predicted by a syntactic parser
        :param target: target framework string
        """
        if conllu is None:
            raise ValueError("conllu is required for tokens and features")
        self.args = Config().args
        self.constraints = CONSTRAINTS.get(graph.framework, Constraints)()
        self.log = []
        self.finished = False
        self.graph = graph
        self.conllu = conllu
        self.framework = target or self.graph.framework
        self.labeled = bool(graph and graph.nodes)
        self.stack = []
        self.buffer = deque()
        self.nodes = []
        self.heads = set()
        self.need_label = self.need_property = self.need_attribute = self.last_edge = None  # Which edge/node is next
        graph_nodes = list(self.graph.nodes)  # Copy list of nodes before adding virtual root and terminals to graph
        self.root = StateNode(ROOT_ID, is_root=True, orig_node=self.graph.add_node(ROOT_ID))  # Virtual root for tops
        self.terminals = []
        for i, conllu_node in enumerate(self.conllu.nodes):
            text = conllu_node.label
            new_conllu_node = self.graph.add_node(
                label=text, anchors=conllu_node.anchors, properties=conllu_node.properties, values=conllu_node.values)
            self.terminals.append(StateNode(i, text=text, orig_node=new_conllu_node))  # Virtual node for tokens
        for node in graph_nodes:
            if node.is_top:
                self.graph.add_edge(ROOT_ID, node.id, ROOT_LAB)
            if node.anchors:
                anchors = StateNode.anchors(node)
                for terminal in self.terminals:
                    if anchors & terminal.orig_anchors:
                        self.graph.add_edge(node.id, terminal.orig_node.id, ANCHOR_LAB)
        self.stack.append(self.root)
        self.buffer += self.terminals
        self.actions = []  # History of applied actions
        self.type_validity_cache = {}

    def create_graph(self):
        """
        Create final graph from temporary representation
        :return: Graph created from self.nodes
        """
        Config().print("Creating %s graph %s from state..." % (self.framework, self.graph.id), level=2)
        graph = Graph(self.graph.id, self.framework)
        graph.input = self.graph.input
        for node in self.nodes:
            if node.text is None and not node.is_root:
                if node.label is None and requires_node_labels(self.framework):
                    node.label = DEFAULT_LABEL
                properties, values = zip(*node.properties.items()) if node.properties else (None, None)
                node.node = graph.add_node(int(node.id), label=node.label, properties=properties, values=values)
        for node in self.nodes:
            for edge in node.outgoing:
                if node.is_root:
                    edge.child.node.is_top = True
                elif edge.child.text is not None:
                    if requires_anchors(self.framework):
                        if node.anchors is None:
                            node.anchors = []
                        node.anchors += edge.child.orig_node.anchors()
                else:
                    attributes, values = zip(*edge.attributes.items()) if edge.attributes else (None, None)
                    graph.add_edge(int(edge.parent.id), int(edge.child.id), edge.lab,
                                   attributes=attributes, values=values)
        return graph

    def is_valid_action(self, action):
        """
        :param action: action to check for validity
        :return: is the action (including tag) valid in the current state?
        """
        valid = self.type_validity_cache.get(action.type_id)
        if valid is None:
            try:
                self.check_valid_action(action)
                valid = True
            except InvalidActionError as e:
                valid = False
                if e.is_type:
                    self.type_validity_cache[action.type_id] = valid
        return valid

    def check_valid_action(self, action, message=False):
        """
        Raise InvalidActionError if the action is invalid in the current state
        :param action: action to check for validity
        :param message: whether to add an informative message to the thrown exception
        """
        def _check_possible_node():
            self.check(self.node_ratio() < self.args.max_node_ratio,
                       message and "Non-terminals/terminals ratio: %.3f" % self.args.max_node_ratio, is_type=True)
            for head in self.heads:
                self.check(head.height <= self.args.max_height,
                           message and "Graph height: %d" % self.args.max_height, is_type=True)

        def _check_possible_parent(node, t):
            self.check(node.text is None, message and "Terminals may not have children: %s" % node.text, is_type=True)
            if self.args.constraints and t is not None:
                for rule in self.constraints.tag_rules:
                    violation = rule.violation(node, t, Direction.outgoing, message=message)
                    self.check(violation is None, violation)
                self.check(self.constraints.allow_parent(node, t),
                           message and "%s may not be a '%s' parent (currently %s)" % (
                               node, t, ", ".join(map(str, node.outgoing)) or "childless"))

        def _check_possible_child(node, t):
            if self.args.constraints and t is not None:
                for rule in self.constraints.tag_rules:
                    violation = rule.violation(node, t, Direction.incoming, message=message)
                    self.check(violation is None, violation)
                self.check(self.constraints.allow_child(node, t),
                           message and "%s may not be a '%s' child (currently %s, %s)" % (
                               node, t, ", ".join(map(str, node.incoming)) or "parentless",
                               ", ".join(map(str, node.outgoing)) or "childless"))

        def _check_possible_edge(p, c, t):
            _check_possible_parent(p, t)
            _check_possible_child(c, t)
            if self.args.constraints and t is not None:
                if p is self.root:
                    self.check(self.constraints.top_level_allowed is None or not t or
                               t in self.constraints.top_level_allowed, message and "Root may not have %s edges" % t)
                else:
                    self.check(self.constraints.top_level_only is None or
                               t not in self.constraints.top_level_only, message and "Only root may have %s edges" % t)
            self.check(self.constraints.allow_root_terminal_children or p is not self.root or c.text is None,
                       message and "Terminal child '%s' for root" % c, is_type=True)
            if self.constraints.multigraph:  # Nodes may be connected by more than one edge
                edge = StateEdge(p, c, t)
                self.check(self.constraints.allow_edge(edge), message and "Edge not allowed: %s (currently: %s)" % (
                               edge, ", ".join(map(str, p.outgoing)) or "childless"))
            else:  # Simple graph, i.e., no more than one edge between the same pair of nodes
                self.check(c not in p.children, message and "%s is already %s's child" % (c, p), is_type=True)
            self.check(p not in c.descendants, message and "Detected cycle by edge: %s->%s" % (p, c), is_type=True)

        def _check_possible_label():
            self.check(requires_node_labels(self.graph.framework), message and "Node labels disabled", is_type=True)
            try:
                node = self.stack[-action.tag]
            except IndexError:
                node = None
            self.check(node is not None, message and "Labeling invalid node %s when stack size is %d" % (
                action.tag, len(self.stack)))
            self.check(node.label is not None, message and "Labeling already-labeled node: %s" % node, is_type=True)
            self.check(node.text is None, message and "Terminals do not have labels: %s" % node, is_type=True)

        def _check_possible_property():
            self.check(requires_node_properties(self.graph.framework), message and "Node properties disabled",
                       is_type=True)
            try:
                node = self.stack[-action.tag]
            except IndexError:
                node = None
            self.check(node is not None, message and "Setting property on invalid node %s when stack size is %d" % (
                action.tag, len(self.stack)))
            self.check(node.text is None, message and "Terminals do not have properties: %s" % node, is_type=True)

        def _check_possible_attribute():
            self.check(requires_edge_attributes(self.graph.framework), message and "Edge attributes disabled",
                       is_type=True)
            self.check(self.last_edge is not None, message and "Setting attribute on edge when no edge exists",
                       is_type=True)

        if self.args.constraints:
            self.check(self.constraints.allow_action(action, self.actions),
                       message and "Action not allowed: %s " % action + (
                           ("after " + ", ".join("%s" % a for a in self.actions[-3:])) if self.actions else "as first"))
        if action.is_type(Actions.Finish):
            self.check(not self.buffer, "May only finish at the end of the input buffer", is_type=True)
            if self.args.swap:  # Without swap, the oracle may be incapable even of single action
                self.check(self.root.outgoing or all(n is self.root or n.text for n in self.nodes),
                           message and "Root has no child at parse end", is_type=True)
            for n in self.nodes:
                self.check(not self.args.require_connected or n is self.root or n.text or
                           n.incoming, message and "Non-terminal %s has no parent at parse end" % n, is_type=True)
                self.check(not requires_node_labels(self.framework) or n.text or n.label is not None,
                           message and "Non-terminal %s has no label at parse end" % n, is_type=True)
        else:
            self.check(self.action_ratio() < self.args.max_action_ratio,
                       message and "Actions/terminals ratio: %.3f" % self.args.max_action_ratio, is_type=True)
            if action.is_type(Actions.Shift):
                self.check(self.buffer, message and "Shifting from empty buffer", is_type=True)
            elif action.is_type(Actions.Label):
                _check_possible_label()
            elif action.is_type(Actions.Property):
                _check_possible_property()
            elif action.is_type(Actions.Attribute):
                _check_possible_attribute()
            else:   # Unary actions
                self.check(self.stack, message and "%s with empty stack" % action, is_type=True)
                s0 = self.stack[-1]
                if action.is_type(Actions.Reduce):
                    if s0 is self.root:
                        self.check(self.root.label is not None or not requires_node_labels(self.framework),
                                   message and "Reducing root without label", is_type=True)
                    elif not s0.text:
                        self.check(not self.args.require_connected or s0.incoming,
                                   message and "Reducing parentless non-terminal %s" % s0, is_type=True)
                        self.check(not self.constraints.required_outgoing or
                                   s0.outgoing_labs.intersection(self.constraints.required_outgoing),
                                   message and "Reducing non-terminal %s without %s edge" % (
                                       s0, self.constraints.required_outgoing), is_type=True)
                    self.check(not requires_node_labels(self.framework) or s0.text or s0.label is not None,
                               message and "Reducing non-terminal %s without label" % s0, is_type=True)
                elif action.is_type(Actions.Swap):
                    # A regular swap is possible since the stack has at least two elements;
                    # A compound swap is possible if the stack is longer than the distance
                    distance = action.tag or 1
                    self.check(1 <= distance < len(self.stack), message and "Invalid swap distance: %d" % distance)
                    swapped = self.stack[-distance - 1]
                    # To prevent swap loops: only swap if the nodes are currently in their original order
                    self.check(self.swappable(s0, swapped),
                               message and "Already swapped nodes: %s (swap index %g) <--> %s (swap index %g)"
                               % (swapped, swapped.swap_index, s0, s0.swap_index))
                else:
                    pct = self.get_parent_child_tag(action)
                    self.check(pct, message and "%s with len(stack) = %d" % (action, len(self.stack)), is_type=True)
                    parent, child, tag = pct
                    if parent is None:
                        _check_possible_child(child, tag)
                        _check_possible_node()
                    elif child is None:
                        _check_possible_parent(parent, tag)
                        _check_possible_node()
                    else:  # Binary actions
                        _check_possible_edge(parent, child, tag)

    @staticmethod
    def swappable(right, left):
        return left.swap_index < right.swap_index

    def is_valid_label(self, label):
        """
        :param label: label to check for validity
        :return: is the label valid in the current state?
        """
        try:
            self.check_valid_label(label)
        except InvalidActionError:
            return False
        return True

    def check_valid_label(self, label, message=False):
        if self.args.constraints and label is not None:
            valid = self.constraints.allow_label(self.need_label, label)
            self.check(valid, message and "May not label %s as %s: %s" % (self.need_label, label, valid))

    def is_valid_property_value(self, property_value):
        """
        :param property_value: property_value to check for validity
        :return: is the property_value valid in the current state?
        """
        try:
            self.check_valid_property_value(property_value)
        except InvalidActionError:
            return False
        return True

    def check_valid_property_value(self, property_value, message=False):
        if self.args.constraints and property_value is not None:
            valid = self.constraints.allow_property_value(self.need_property, property_value)
            self.check(valid, message and "May not set property value for %s to %s: %s" % (
                self.need_property, property_value, valid))

    def is_valid_attribute_value(self, attribute_value):
        """
        :param attribute_value: attribute_value to check for validity
        :return: is the attribute_value valid in the current state?
        """
        try:
            self.check_valid_attribute_value(attribute_value)
        except InvalidActionError:
            return False
        return True

    def check_valid_attribute_value(self, attribute_value, message=False):
        if self.args.constraints and attribute_value is not None:
            valid = self.constraints.allow_attribute_value(self.need_attribute, attribute_value)
            self.check(valid, message and "May not set attribute value for %s to %s: %s" % (
                self.need_attribute, attribute_value, valid))

    def is_valid_annotation(self, key=None):
        """
        :param key: one of NODE_LABEL_KEY, NODE_PROPERTY_KEY, EDGE_ATTRIBUTE_KEY or None
        :return function to check validity of values
        """
        if key is None:
            return self.is_valid_action
        if key == NODE_LABEL_KEY:
            return self.is_valid_label
        if key == NODE_PROPERTY_KEY:
            return self.is_valid_property_value
        if key == EDGE_ATTRIBUTE_KEY:
            return self.is_valid_attribute_value
        raise ValueError("Invalid key: " + str(key))

    @staticmethod
    def check(condition, *args, **kwargs):
        if not condition:
            raise InvalidActionError(*args, **kwargs)

    # noinspection PyTypeChecker
    def transition(self, action):
        """
        Main part of the parser: apply action given by oracle or classifier
        :param action: Action object to apply
        """
        action.apply()
        self.log = []
        pct = self.get_parent_child_tag(action)
        if pct:
            parent, child, tag = pct
            if parent is None:
                parent = action.node = self.add_node(orig_node=action.orig_node)
            if child is None:
                child = action.node = self.add_node(orig_node=action.orig_node)
            action.edge = self.add_edge(StateEdge(parent, child, tag, orig_edge=action.orig_edge))
            if action.node:
                self.buffer.appendleft(action.node)
        elif action.is_type(Actions.Shift):  # Push buffer head to stack; shift buffer
            self.stack.append(self.buffer.popleft())
        elif action.is_type(Actions.Label):
            self.need_label = self.stack[-action.tag]  # The parser is responsible to choose a label and set it
        elif action.is_type(Actions.Property):
            self.need_property = self.stack[-action.tag]  # The parser is responsible to choose a property and set it
        elif action.is_type(Actions.Attribute):
            self.need_attribute = self.last_edge  # The parser is responsible to choose an attribute and set it
        elif action.is_type(Actions.Reduce):  # Pop stack (no more edges to create with this node)
            self.stack.pop()
        elif action.is_type(Actions.Swap):  # Place second (or more) stack item back on the buffer
            distance = action.tag or 1
            s = slice(-distance - 1, -1)
            self.log.append("%s <--> %s" % (", ".join(map(str, self.stack[s])), self.stack[-1]))
            self.buffer.extendleft(reversed(self.stack[s]))  # extendleft reverses the order
            del self.stack[s]
        elif action.is_type(Actions.Finish):  # Nothing left to do
            self.finished = True
        else:
            raise ValueError("Invalid action: %s" % action)
        action.index = len(self.actions)
        self.actions.append(action)
        self.type_validity_cache = {}

    def add_node(self, **kwargs):
        """
        Called during parsing to add a new StateNode (not graph.Node) to the temporary representation
        :param kwargs: keyword arguments for StateNode()
        """
        node = StateNode(len(self.nodes), swap_index=self.calculate_swap_index(), **kwargs)
        self.nodes.append(node)
        self.heads.add(node)
        self.log.append("node: %s (swap_index: %g)" % (node, node.swap_index))
        return node

    def calculate_swap_index(self):
        """
        Update a new node's swap index according to the nodes before and after it.
        Usually the swap index is just the index, i.e., len(self.nodes).
        If the buffer is not empty and its head is not a terminal, it means that it is a non-terminal created before.
        In that case, the buffer head's index will be lower than the new node's index, so the new node's swap index will
        be the arithmetic mean between the previous node (stack top) and the next node (buffer head).
        Then, in the validity check on the SWAP action, we will correctly identify this node as always having appearing
        before the current buffer head. Otherwise, we would prevent swapping them even though it should be valid
        (because they have never been swapped before).
        """
        if self.buffer:
            b0 = self.buffer[0]
            if self.stack and (b0.text is not None or b0.swap_index <= len(self.nodes)):
                s0 = self.stack[-1]
                return (s0.swap_index + b0.swap_index) / 2
        return None

    def add_edge(self, edge):
        edge.add()
        self.heads.discard(edge.child)
        self.log.append("edge: %s" % edge)
        self.last_edge = edge
        return edge
    
    PARENT_CHILD = (
        (Actions.LeftEdge, (-1, -2)),
        (Actions.RightEdge, (-2, -1)),
        (Actions.Node, (None, -1)),
        (Actions.Implicit, (-1, None)),
    )

    def get_parent_child_tag(self, action):
        try:
            for action_type, indices in self.PARENT_CHILD:
                if action.is_type(action_type):
                    parent, child = [None if i is None else self.stack[i] for i in indices]
                    break
            else:
                return None
            return parent, child, action.tag
        except IndexError:
            return None

    def assign_node_label(self, label):
        assert self.need_label is not None, "Called assign_node_label() when need_label is None"
        assert label is not None, "Labeling node %s with None label" % self.need_label
        self.need_label.label = label
        self.log.append("label: %s" % self.need_label)
        self.type_validity_cache = {}
        self.need_label = None

    def assign_node_property_value(self, property_value):
        assert self.need_property is not None, "Called assign_node_property() when need_property is None"
        assert property_value is not None, "Assigning node %s with None property-value pair" % self.need_property
        prop, value = property_value.split("=")
        self.need_property.properties[prop] = value
        self.log.append("property: %s" % self.need_property)
        self.type_validity_cache = {}
        self.need_property = None

    def assign_edge_attribute_value(self, attribute_value):
        assert self.need_attribute is not None, "Called assign_edge_attribute() when need_attribute is None"
        assert attribute_value is not None, "Assigning edge %s with None attribute-value pair" % self.need_attribute
        attrib, value = attribute_value.split("=")
        self.need_attribute.attributes[attrib] = value
        self.log.append("attribute: %s" % self.need_attribute)
        self.type_validity_cache = {}
        self.need_attribute = None

    def node_ratio(self):
        return (len(self.nodes) / len(self.terminals) - 1) if self.terminals else 0

    def action_ratio(self):
        return (len(self.actions) / len(self.terminals)) if self.terminals else 0

    def str(self, sep):
        return "stack: [%-20s]%sbuffer: [%s]" % (" ".join(map(str, self.stack)), sep,
                                                 " ".join(map(str, self.buffer)))

    def __str__(self):
        return self.str(" ")

    def __eq__(self, other):
        return self.stack == other.stack and self.buffer == other.buffer and \
               self.nodes == other.nodes

    def __hash__(self):
        return hash((tuple(self.stack), tuple(self.buffer), tuple(self.nodes)))

import networkx as nx
import sys


class Node:

    def __init__(
        self,
        label=None,
        children=None,
        fv=None,
        lv=None,
        parent=None,
        ):
        self.label = label
        self.children = children
        self.fv = fv
        self.lv = lv
        self.parent = parent


class PartitionNode:

    def __init__(self, P):
        self.P = P
        self.next = None
        self.prev = None


class PartitionList:

    def __init__(self, head):
        self.head = head

    def split(self, arr, Prt):
        if self.head == Prt:
            self.head = arr[0]
            arr[-1].next = Prt.next
            if Prt.next:
                Prt.next.prev = arr[-1]
        else:

            Prt.prev.next = arr[0]
            arr[0].prev = Prt.prev
            arr[-1].next = Prt.next
            if Prt.next:

                Prt.next.prev = arr[-1]

    def __str__(self):
        res = []
        h = self.head
        while h:
            res.append(h.P)
            h = h.next
        return str(res)


def getEdgesAdjL(line):

    (nodea, nodesb) = line.split(' | ')
    nodesb = nodesb.split(',')
    edges = [(nodea, nodeb) for nodeb in nodesb]
    return edges


def readAdjFile(F):
    Gp = nx.Graph()
    for line in F:

        line = line.split('\n')[0]
        edges = getEdgesAdjL(line)
        Gp.add_edges_from(edges)
    return Gp


def decomposer(adjl):
    G = readAdjFile(adjl)
    nodelist = set(G.nodes)
    init = PartitionNode(nodelist)
    Partition = PartitionList(init)
    dummy = PartitionNode(nodelist)
    dummy_part = PartitionList(dummy)
    return (decompose(
        list(init.P)[-1],
        dummy,
        G,
        Partition,
        dummy_part,
        init,
        ), G)


def min_max(A, B):
    if len(A) > len(B):
        return (B, A)
    return (A, B)


def decompose(
    v,
    X,
    G,
    Partition,
    cur,
    main,
    ):
    Gs = G.subgraph(X.P)
    Nv = PartitionNode(set(Gs.neighbors(v)))
    center = PartitionNode(set([v]))
    Nv_bar = PartitionNode(X.P.difference(Nv.P).difference(center.P))

    if len(Nv_bar.P):
        Nv_bar.next = center
        center.prev = Nv_bar
    if len(Nv.P):
        center.next = Nv
        Nv.prev = center
    A = [i for i in [Nv_bar, center, Nv] if len(i.P) > 0]

    cur.split(A, X)

    (smaller, larger) = min_max(Nv_bar.P, Nv.P)

    S = []
    K = []
    L = []
    if len(smaller) > 0:
        L.append(smaller)
    if len(larger) > 0:
        K.append(larger)
    between = False

    while len(K) + len(L) > 0:

        if len(L) > 0:
            S = L.pop()
        else:
            S = set(list(K.pop())[0])

        cnt = 0
        for x in S:

            cnt += 1

            part = cur.head
            while part:

                y = part.P

                if x in y or v in y:
                    between = not between
                    part = part.next
                    continue
                Y1 = set()
                Nadj_bar = set()
                for node in y:
                    if G.has_edge(node, x):
                        Y1.add(node)
                    else:
                        Nadj_bar.add(node)

                if len(Y1) > 0 and len(Y1) < len(y):
                    if between:
                        left = PartitionNode(Y1)
                        right = PartitionNode(Nadj_bar)
                    else:
                        left = PartitionNode(Nadj_bar)
                        right = PartitionNode(Y1)
                    left.next = right
                    right.prev = left
                    next_part = part.next
                    cur.split([left, right], part)
                    part = next_part

                    (Ymin, Ymax) = min_max(Y1, Nadj_bar)
                    if y in L:
                        L.remove(y)
                        L += [Ymin, Ymax]
                    else:
                        L.append(Ymin)
                        flag = True
                        for (i, z) in enumerate(K):
                            if z == y:
                                K[i] = Ymax
                                flag = False
                                break
                        if flag:
                            K.append(Ymax)
                else:

                    part = part.next
    modarray = []
    mod = cur.head
    while mod:
        modarray.append(mod)
        mod = mod.next

    Partition.split(modarray, main)
    #print('Partition', Partition)
    module = Partition.head
    while module:
        next_module = module.next
        if len(module.P) > 1:
            temp = PartitionNode(module.P)
            cur = PartitionList(temp)
            decompose(
                list(module.P)[-1],
                temp,
                G,
                Partition,
                cur,
                module,
                )
        module = next_module
    res = []
    cu = Partition.head
    while cu:
        for vertex in cu.P:
            res.append(vertex)
        cu = cu.next
    #print ('Factoring Permutation ->', res)
    return res


def paranthesize(perm, G):
    n = len(perm)
    lc = [0 for i in range(n)]
    rc = [0 for i in range(n)]
    vl = [-1 for i in range(n)]
    vr = [-1 for i in range(n)]
    perm_map = {}
    for i in range(n):
        perm_map[perm[i]] = i
    for (i, v) in enumerate(perm):
        for adj in G.adj[v]:

            pos = perm_map[adj]
            if pos > i:
                if not pos == i + 1 and not G.has_edge(v, perm[pos
                        - 1]) and vl[pos - 1] < 0:
                    lc[i] += 1
                    vl[pos - 1] = i
                    rc[pos - 1] += 1
                if pos < n - 1:
                    if not G.has_edge(v, perm[pos + 1]) and vl[pos] < 0:
                        lc[i] += 1
                        vl[pos] = i
                        rc[pos] += 1
    for (i, v) in reversed(list(enumerate(perm))):
        for adj in G.adj[v]:
            pos = perm_map[adj]
            if pos < i:
                if not pos == i - 1 and not G.has_edge(v, perm[pos
                        + 1]) and vr[pos] < 0:
                    rc[i] += 1
                    vr[pos] = i
                    lc[pos + 1] += 1
                if pos > 0:
                    if not G.has_edge(v, perm[pos - 1]) and vr[pos - 1] \
                        < 0:
                        rc[i] += 1
                        vr[pos - 1] = i
                        lc[pos] += 1
    res1 = []

    for (i, v) in enumerate(perm):
        for te in range(lc[i]):
            res1.append('(')
        res1.append(v)
        for te in range(rc[i]):
            res1.append(')')
    #print(''.join(res1))
    remove_dummies(
        perm,
        G,
        vl,
        vr,
        lc,
        rc,
        )
    res = []
    for (i, v) in enumerate(perm):
        while lc[i] > 0:
            res.append('(')
            lc[i] -= 1
        res.append(v)
        while rc[i] > 0:
            res.append(')')
            rc[i] -= 1

    return (res, vl, vr)


def remove_dummies(
    perm,
    G,
    vl,
    vr,
    lc,
    rc,
    ):
    stack = []
    n = len(perm)
    for i in range(n):
        for j in range(lc[i]):
            stack.append(i)
        for j in range(rc[i]):
            cur = stack.pop()
            #print(cur,i)
            if cur == i:
                lc[cur] -= 1
                rc[cur] -= 1
            if cur < i:
                fleft = min(vl[cur:i])
                fright = max(vr[cur:i])
                if (fleft >= 0 and fleft< cur) \
                    or fright > i:

                    lc[cur] -= 1
                    rc[i] -= 1


def perm_to_tree(
    G,
    perm,
    cur,
    i,
    j,
    ):
    if i >= len(perm):
        return (i, j)
    if perm[i] == '(':
        if cur.children:
            cur.children.append(Node(parent=cur))
            (i, j) = perm_to_tree(G, perm, cur.children[-1], i + 1, j)
        else:
            cur.children = []
            cur.children.append(Node(parent=cur))
            (i, j) = perm_to_tree(G, perm, cur.children[-1], i + 1, j)
    elif perm[i] != ')':
        if cur.children:
            cur.children.append(Node(fv=j, lv=j, label=j, parent=cur))
        else:
            cur.children = []
            cur.children.append(Node(fv=j, lv=j, label=j, parent=cur))
        (i, j) = perm_to_tree(G, perm, cur, i + 1, j + 1)
    else:
        cur = cur.parent
        (i, j) = perm_to_tree(G, perm, cur, i + 1, j)

    return (i, j)


def label_tree(G, cur, perm):
    if cur.children:
        for node in cur.children:
                label_tree(G, node, perm)

        if(len(cur.children)>=2):
            
            q = []
            for node in cur.children:
                q.append(perm[node.fv])
            first = cur.children[0].fv
            last = cur.children[-1].lv
            Gp = G.subgraph(q)
            out = 0
            for v in q:
                if Gp.degree[v] == len(q) - 1:
                    out += 1
                elif Gp.degree[v] == 0:
                    out -= 1
            if out == len(q):
                cur.label = 'series'
            elif out == -len(q):
                cur.label = 'parallel'
            else:
                cur.label = 'prime'
            cur.fv = first
            cur.lv = last
        else:

            if(cur.children[0].children):
                cur.label = cur.children[0].label
                cur.fv = cur.children[0].fv
                cur.lv = cur.children[0].lv
                cur.children = cur.children[0].children
            else:
                cur.fv = cur.children[0].fv
                cur.lv = cur.children[0].lv
                cur.children = None
                cur.label = None

def get_merged_nodes(
    G,
    cur,
    perm,
    vl,
    vr,
    ):
    if cur.children:
        for node in cur.children:
            get_merged_nodes(G, node, perm, vl, vr)
        twins = []
        if(cur.label == 'prime'):
            for (i, node) in enumerate(cur.children):
                if (vl[node.lv] == -1 or vl[node.lv] >= int(node.fv)) \
                    and (vr[node.lv] == -1 or vr[node.lv]
                         <= int(node.lv)) and i + 1 \
                    < len(cur.children):
                    print(perm[node.lv],vl[node.lv],perm[node.lv],vr[node.lv])
                    if len(twins) == 0:
                        twins += [node, cur.children[i + 1]]
                    else:
                        twins += [cur.children[i + 1]]
                else:
                    true_twins = 0
                    false_twins = 0
                    if len(twins) > 0:
                        for tw in twins:
                            print(tw.fv,tw.lv)

                        for (j, twin) in enumerate(twins):
                            if j + 1 < len(twins) and G.has_edge(twin.fv,
                                    twins[j + 1].fv):
                                true_twins += 1
                            elif j + 1 < len(twins):
                                false_twins += 1
                    if true_twins + 1 == len(twins) or false_twins + 1 \
                        == len(twins):
                        new_Node = Node(fv=twins[0].fv, lv=twins[-1].lv,
                                        children=twins.copy())
                        for child in new_Node.children:
                            child.parent = new_Node
                        for k in range(i, i - len(twins), -1):
                            cur.children.pop(k)
                        cur.children.insert(i, new_Node)
                        if true_twins + 1 == len(twins):
                            new_Node.label = 'series'
                        else:
                            new_Node.label = 'parallel'

                    twins = []


def create_tree(adj_file):
    (a, b) = decomposer(filek)
    filek.close()
    (ans, lmap, rmap) = paranthesize(a, b)
    print("Paranthesized Permutation",ans)
    root = Node()
    perm_to_tree(b, ans, root, 0, 0)
    label_tree(b, root, a)
    #print(root.children)
    get_merged_nodes(b, root, a, lmap, rmap)

    return ''.join(to_newick(root,[';'],a))

def to_newick(root,res,perm):
	if(root):
		if(root.children):
			if(root.label=='prime'):
				res.append('*')
			if(root.label=='series'):
				res.append('S')
			if(root.label=='parallel'):
				res.append('P')
			res.append(')')
			for i,child in enumerate(root.children):
				to_newick(child,res,perm)
				if(i!=len(root.children)-1):
					res.append(',')
			res.append('(')
		else:
			res.append(perm[root.fv])
			
	return res[::-1]





if __name__ == '__main__':
    filek = open(sys.argv[1], 'r')
    Tree = create_tree(filek)
    print("Tree Created:")
    print(Tree)

    #print (root.children)

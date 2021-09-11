import networkx as nx
import sys
class PartitionNode:
	def __init__(self,P):
		self.P = P
		self.next = None
		self.prev = None
it = 0
class PartitionList:
	def __init__(self,head):
		self.head = head

	def split(self,arr,Prt):
		if(self.head==Prt):
			self.head = arr[0]
			arr[-1].next = Prt.next
		else:
			Prt.prev.next = arr[0]
			arr[0].prev = Prt.prev
			arr[-1].next = Prt.next
			if(Prt.next):
				Prt.next.prev = arr[-1]
	def __str__(self):
		res = []
		h = self.head
		while(h):
			res.append(h.P)
			h = h.next
		return str(res)
		


def getEdgesAdjL(line):
	
		nodea,nodesb=line.split(" | ")
		nodesb=nodesb.split(",")
		edges=[(nodea,nodeb) for nodeb in nodesb]
		return edges

def readAdjFile(F):
	Gp = nx.Graph()
	for line in F:
		
		line = line.split('\n')[0]
		edges=getEdgesAdjL(line)
		Gp.add_edges_from(edges)
	return Gp


def decomposer(adjl):
	G =  readAdjFile(adjl)
	nodelist = set(G.nodes)
	init = PartitionNode(nodelist)
	Partition = PartitionList(init)
	dummy = PartitionNode(nodelist)
	dummy_part = PartitionList(dummy)
	return (decompose(list(init.P)[-1],dummy,G,Partition,dummy_part,0,init),G)
	
	
def min_max(A,B):
	if(len(A)>len(B)):
		return (B,A)
	return (A,B)

def decompose(v,X,G,Partition,cur,i,main):
	
	Gs = G.subgraph(X.P)
	Nv = PartitionNode(set(Gs.neighbors(v)))
	center = PartitionNode(set([v]))
	Nv_bar = PartitionNode(X.P.difference(Nv.P).difference(center.P))

	if(len(Nv_bar.P)):
		Nv_bar.next = center
		center.prev = Nv_bar
	if(len(Nv.P)):
		center.next = Nv
		Nv.prev = center
	A = [i for i in [Nv_bar,center,Nv] if len(i.P)>0]
	#print("Center",center.P)
	#print("Neigh",Nv.P)
	#print("Non",Nv_bar.P)
	cur.split(A,X)
	#print("cur",cur)

	smaller,larger = min_max(Nv_bar.P,Nv.P)
	S = []
	K = []
	L = []
	if(len(smaller)>0):
		L.append(smaller)
	if(len(larger)>0):
		K.append(larger)
	between = False
	while(len(K) + len(L) > 0):
		#print("L",L,"K",K)

		if(len(L)>0):
			S = L.pop()
		else:
			S = set(list(K.pop())[0])
		#print(S)
		for x in S:
			part = cur.head
			while(part):
				#print(part.P)


				y = part.P
				
				if((x in y) or  (v in y)):
					between =  not between
					part = part.next
					continue
				Nadj = set(Gs.neighbors(x))
				Y1 = Nadj.intersection(y)
				if(len(Y1)> 0 and len(Y1)<len(y)):
					Nadj_bar = set(y.difference(Nadj))
					if(between):
						left = PartitionNode(Y1)
						right = PartitionNode(Nadj_bar)
					else:
						left = PartitionNode(Nadj_bar)
						right = PartitionNode(Y1)
					left.next = right
					right.prev = left
					next_part = part.next
					cur.split([left,right],part)
					part = next_part

					Ymin,Ymax = min_max(Y1,Nadj_bar)
					if y in L:
						L.remove(y)
						L+=[Ymin,Ymax]
					else:
						L.append(Ymin)
						for (i,y) in enumerate(K):
							if(y==K[i]):
								K[i] = Ymax
								break
						else:
							K.append(Ymax)
				else:
					part = part.next
	modarray = []
	mod = cur.head
	while(mod):
			modarray.append(mod)
			mod = mod.next
	#print([lm.P for lm in modarray])
	Partition.split(modarray,main)
	mod = Partition.head
	while(mod):
		#print(mod.P,end = "")
		mod = mod.next
	#print("")

	module = Partition.head
	while(module and it<1):
		next_module = module.next
		if(len(module.P)>1):
			temp = PartitionNode(module.P)
			cur = PartitionList(temp)
			decompose(list(module.P)[-1],temp,G,Partition,cur,it+1,module)
		module = next_module
	res = []
	cu = Partition.head
	while(cu):
		for vertex in cu.P:
			res.append(vertex)
		cu = cu.next
	print(res)
	return "".join(res)

def paranthesize(perm,G):
	n = len(perm)
	lc = [0 for i in range(n)]
	rc = [0 for i in range(n)]
	vl = [-1 for i in range(n)]
	vr = [-1 for i in range(n)]
	lc[0] = 1
	rc[n-1] = 1
	perm_map = {}
	for i in range(n):
		perm_map[perm[i]] = i
		#print(perm[i])
	for i,v in enumerate(perm):
		for adj in G.adj[v]:
			#print(v,adj)
			pos = perm_map[adj]
			if(pos>i):
				if((not pos == i+1) and (not G.has_edge(v,perm[pos-1])) and (vl[pos-1]<0)):
					lc[i]+=1
					vl[pos-1] = i
					rc[pos-1]+=1
				if(pos<n-1):
					if(not G.has_edge(v,perm[pos+1]) and (vl[pos]<0)):
						lc[i]+=1
						vl[pos] = i
						rc[pos]+=1
	for i,v in reversed(list(enumerate(perm))):
		for adj in G.adj[v]:
			#print(v,adj)
			pos = perm_map[adj]
			if(pos<i):
				if((not pos == i-1) and (not G.has_edge(v,perm[pos+1])) and (vr[pos]<0)):
					rc[i]+=1
					vr[pos] = i
					lc[pos]+=1
				if(pos>0):
					if(not G.has_edge(v,perm[pos-1]) and (vr[pos-1]<0)):
						rc[i]+=1
						vr[pos-1] = i
						lc[pos-1]+=1
	remove_dummies(perm,G,vl,vr,lc,rc)
	#print(lc)
	#print(rc)
	res = ""
	for i,v in enumerate(perm):
		while(lc[i]>0):
			res+='('
			lc[i]-=1
		res+=v
		while(rc[i]>0):
			res+=')'
			rc[i]-=1
	print(res)


def remove_dummies(perm,G,vl,vr,lc,rc):
	res = ""
	for i,v in enumerate(perm):
		for j in range(lc[i]):
			res+='('
		res+=v
		for j in range(rc[i]):
			res+=')'
			
	print(res)
	stack = []
	n = len(perm)
	for i in range(n):
		for j in range(lc[i]):
			stack.append(i)
		for j in range(rc[i]):
			cur = stack.pop()
			if(cur<i):
				left_cutter = min(vl[cur:i])
				right_cutter = max(vr[cur:i])
				if((left_cutter>=0 and left_cutter<cur)  or right_cutter>i):
					print(cur,left_cutter,i,right_cutter)
					lc[cur]-=1
					rc[i]-=1

if __name__ == "__main__":	
	filek = open(sys.argv[1],'r')
	a,b = decomposer(filek)
	ans = paranthesize(a,b)
	print(ans)


						




#G = nx.Graph()
#G.add_edges_from([(1,2),(2,3)])
#A = list(G.nodes)
#B = list(G.nodes)

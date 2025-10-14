from collections import OrderedDict

from bcause.factors import MultinomialFactor
from bcause.factors.values import NumpyStore, BTreeStore
from bcause.factors.values.btreestore import BTreeNode, BTreeStore, BTreeNodeConsecutive, BTreeNodeNonConsecutive
from bcause.factors.values.btreeops import BTreeStoreOperations

import pandas as pd
import numpy as np
from bcause.factors.values.btreestore import BTreeStore
from bcause.models.cmodel import StructuralCausalModel


# different ways of building a node
# Import the model used in WUPES' paper.
m2 = StructuralCausalModel.read("/Users/antoniogonzalezalves/Desktop/model_wupes.bif")
data = pd.read_csv("/Users/antoniogonzalezalves/Desktop/data_wupes.csv")

exovar = "U"

def reshape_value(domain, unshaped_values):
    dom = domain
    shape = [len(d) for d in dom.values()]
    val = np.reshape(unshaped_values, shape)
    return val

dom_W = m2.factors["W"].domain
bt_W = BTreeStore(domain=dom_W, data=reshape_value(dom_W,m2.factors["W"].values), exovar=exovar, is_equation=False)

dom_T = m2.factors["T"].domain
bt_T = BTreeStore(domain=dom_T, data=reshape_value(dom_T,m2.factors["T"].values), exovar=exovar, is_equation=True)

dom_S = m2.factors["S"].domain
bt_S = BTreeStore(domain=dom_S, data=reshape_value(dom_S,m2.factors["S"].values), exovar=exovar, is_equation=True)

endo_domain = dict(H=[0, 1], T=[0, 1], S=[0, 1])
from bcause.util.datautils import to_counts
empirical_prob = to_counts(domains=endo_domain, data=data,normalize=True)
endo_domain = dict(H=["0", "1"], T=["0", "1"], S=["0", "1"])
empirical_tree = BTreeStore(domain=endo_domain, data=reshape_value(endo_domain,empirical_prob.values), is_equation=False)
# restricted_tree = BTreeStoreOperations.restrict(bt_T, {"U":["u0","u4"]})

dom_test = m2.factors["U"].domain
bt_test = BTreeStore(domain=dom_test, data=reshape_value(dom_test,m2.factors["U"].values), is_equation=False)
bt_test.set_data(BTreeStore.var_to_nonconsecutive(bt_test.data, "U"))

T1 = BTreeStoreOperations.multiply_SE(bt_T, bt_S)
test_mult = BTreeStoreOperations.multiply(T1,bt_test)
ttt_test = BTreeStore(data=BTreeStoreOperations._mult_exogenous(T1.data, bt_test.data), domain= T1.domain)
T2 = BTreeStoreOperations.multiply(empirical_tree,T1)
T3 = BTreeStoreOperations.multiply(bt_test,T2)
aa= bt_test.restrict(U=["u0","u1","u2"])
T4 = BTreeStoreOperations.marginalize(test_mult, ["U"])

T5 = BTreeStoreOperations.divide(T2, T4)

T6 = BTreeStoreOperations.marginalize(T5, ["H","T","S"])

T7 = BTreeStoreOperations.multiply(T6, bt_test)

marginalized_exogenous = BTreeStoreOperations.marginalize(T2, ["U"])
aa = BTreeStoreOperations.marginalize(T3, ["H","T","S"])
# print(test_tree.data.summary())

domain = dict(A=["a1", "a2"], B=["b1", "b2", "b3" ,"b4"])
values = [[0.2, .2, 0.5, 0.1], [0.2, 0.2, 0.6 ,0.0]]
# P(B|A) as a numpy table
f = MultinomialFactor(domain, values, left_vars=["B"])


# Operaciones

f * f
f + f
f.R(A="a1", B=["b1","b2"])
f.marginalize("A")


# data store

f.store
type(f.store)

# each store will have a builder function
f.store.builder(domain=domain, data=values)


# P(B|A) as a binary tree
f = MultinomialFactor(domain, values, left_vars=["B"], vtype="btree")



f.store
type(f.store)

print(f.store.data.summary())





# Use of the BTreeNode class:


variable = "U"
var_domain = ["u1", "u2", "u3", "u4", "u5"]

# different ways of building a node
n1 = BTreeNode.build(variable, var_domain, 0.2, 0.4, left_states=["u1", "u2"])
n2 = BTreeNode.build(variable, var_domain, 0.2, 0.4, right_states=["u4", "u5"])
n3 = BTreeNode.build(variable, var_domain, 0.4, 0.3, end_left_exclusive=1)



# a non terminal node
nested_nodes = BTreeNode.build("X", ["x1", "x2"], n1, n2)
nested_nodes = BTreeStore(domain={"U":var_domain, "X":["x1","x2"]}, data=nested_nodes)
# restricted_tree = BTreeStoreOperations.restrict(nested_nodes, {"U":["u3","u4"]})

print(nested_nodes)

domain = dict(A=["a1", "a2"], B=["b1", "b2", "b3", "b4"])
new_var_order = ["B", "A"]
# complete vars
new_dom = dict([(v, domain[v]) for v in new_var_order])
data = [[0.2, .2, 0.5, 0.1], [0.2, 0.2, 0.6, 0.0]]

bt = BTreeStore(domain, data)
print(bt.data.summary())


'''
0) Entender el código anterior
'''

'''
1) Implementar la clase BTreeNodeNonConsecutive. Observaciones
    - el constructor def __init__(self, variable: Hashable, var_domain: List, left_child, right_child, left_states, right_states)
    - Parecida a BTreeNodeConsecutive
    - Internamente puede almacena 2 conjuntos (tipo set) _left_states y _right_states. En ese caso no haría falta almacenar var_domain.
    - Se utiliza el tipo set porque la operación de pertenencia en listas es más eficiente.
    - El hecho de que ahora permitimos nodos con particiones no consicutivas es una novedad
    que habrá que explicar en el paper.
    - Los hijos pueden ser un valor numérico u otro nodo
    - Ejemplo de uso:
    
'''

var_domain = ["u1", "u2", "u3"]


n = BTreeNode.build(variable = "U", var_domain=var_domain, left_child=0.5, right_child=0.25, left_states=["u2"], consecutive=False)
assert set(n.left_states) == set(["u2"])
assert set(n.right_states) == set(["u1","u3"])
assert n.is_on_left("u2") == True
assert n.is_on_right("u1") == True
assert n.left_child == 0.5
assert n.right_child == 0.25

'''
2) En la función BTreeNode.build(..., consecutive=False), implementar la parte del if correspondiente a
consecutive=False. Internamente llama a BTreeNodeNonConsecutive. 

Después de implementar esta función probar a crear un árbol con más niveles


'''
variable = "U"
var_domain = ["u1", "u2", "u3"]

# different ways of building a node
n1 = BTreeNode.build(variable, var_domain, 0.2, 0.4, left_states=["u1"], consecutive=False)
n2 = BTreeNode.build(variable, var_domain, 0.2, 0.4, right_states=["u2", "u3"], consecutive=False)
n3 = BTreeNode.build(variable, None, 0.4, 0.3, left_states=["u1"],right_states=["u2", "u3"], consecutive=False)
# a non terminal node
nested_nodes = BTreeNode.build("X", ["x1", "x2"], 0.33, n1, consecutive=False)
nested_nodes_2 = BTreeNode.build("Y",["y1","y2"], left_child=0.1, right_child=nested_nodes, left_states=["y2"], consecutive=False)

'''
3) Implementar BTreeStore._build_from_equation(table, exovar)

    - Similar a BTreeStore._build_from_table construye un árbol a partir de una ecuación. 
    - El árbol siempre tendrá la variable exógena en la base del árbol.
    - Los nodos de la variable exógena serán de tipo BTreeNodeNonConsecutive
    - Los nodos de las variables endógenas serán de tipo BTreeNodeConsecutive.
    - Las hojas siempre serán 0 o 1
    - Después de implementar esto, intentar ver si se ahorra espacio.
'''

from bcause.models.cmodel import StructuralCausalModel
# m = StructuralCausalModel.read("./models/literature/pearl_small.bif")
#
# table = m.factors["S"]
# exovar = "U"
#
# from typing import Dict, List, Iterable, Union, Hashable
# import numpy as np
#
# values = table.values
# domain = table.domain
# if (isinstance(values, Iterable) and not isinstance(values, dict)) or np.isscalar(values):
#     shape = [len(d) for d in domain.values()]
#     if np.ndim(values) == 0:
#         values = [values] * int(np.prod(shape))
#     if np.ndim(values) == 1: values = np.reshape(values, shape)
#
# cosa = NumpyStore(domain=domain, data=values)
# obs = {"T": "0"}
#
# SCM_tree = BTreeStore(domain=domain, data=values,exovar=exovar,is_equation=True)

'''
4) Adaptar las operaciones en BTreeStoreOperations para que funcionen con nodos de tipo BTreeNodeNonConsecutive.

    - En las operaciones binarias se mantiene el tipo de nodo del primer operando.
    - En las operaciones no binarias no se puede dar el caso de que se opere con dos tipos distintos

'''



# Ejemplo btreeops

#restrict_tree = btops.restrict_btreenode(nested_nodes, {"X":"x2"})
# marginalize_tree = btops.marginalize_btreenode(nested_nodes, "U", 3, lambda x, y: x + y)

# domains
A_dom = ["a1", "a2", "a3"]
B_dom = ["b1", "b2"]
C_dom = ["c1", "c2"]

# Bottom: C subtree (c1 -> 45, c2 -> 10)
C_sub = BTreeNode.build(
    variable="C", var_domain=C_dom,
    left_child=45, right_child=10
)  # binary → consecutive by default

# Left B under A=a2: b1 -> C_sub, b2 -> 20
B_left = BTreeNode.build(
    variable="B", var_domain=B_dom,
    left_child=C_sub, right_child=20
)

# Right B under A=a3: b1 -> 25, b2 -> 50
B_right = BTreeNode.build(
    variable="B", var_domain=B_dom,
    left_child=25, right_child=50,consecutive = False
)

# Middle A node (only a2 vs a3)
A_mid = BTreeNode.build(
    variable="A", var_domain=["a2", "a3"],
    left_child=B_left, right_child=B_right
)  # binary → consecutive

# Root: A with non-consecutive split {a1} | {a2,a3}
root = BTreeNode.build(
    variable="A", var_domain=A_dom,
    left_child=30, right_child=A_mid,
    left_states=["a1"], right_states=["a2", "a3"],
    consecutive=False
)

# See it
print(root.summary())

# marginalize_tree_tesis = btops.marginalize_btreenode(root, "B", 2, lambda x, y: x + y)

domain = dict(A=["a1", "a2"], B=["b1", "b2", "b3", "b4"])
new_var_order = ["B", "A"]
# complete vars
new_dom = dict([(v, domain[v]) for v in new_var_order])
data = [[0.2, .2, 0.5, 0.1], [0.2, 0.2, 0.6, 0.0]]

bt = BTreeStore(domain, data)
print(bt.data.summary())

# marginalize_sum = BTreeStoreOperations.marginalize(bt, ["A"])
# marginalize_max = BTreeStoreOperations.maxmarginalize(bt, ["A"])

import pandas as pd
import numpy as np
from bcause.factors.values.btreestore import BTreeStore

# Import the model used in WUPES' paper.
m2 = StructuralCausalModel.read("/Users/antoniogonzalezalves/Desktop/model_wupes.bif")
data = pd.read_csv("/Users/antoniogonzalezalves/Desktop/data_wupes.csv")

exovar = "U"


dom_T = m2.factors["T"].domain
shape_T = [len(d) for d in dom_T.values()]
val_T = np.reshape(m2.factors["T"].values, shape_T)
bt_T = BTreeStore(domain=dom_T, data=val_T, exovar=exovar, is_equation=True)

dom_S = m2.factors["S"].domain
shape_S = [len(d) for d in dom_S.values()]
val_S = np.reshape(m2.factors["S"].values, shape_S)
bt_S = BTreeStore(domain=dom_S, data=val_S, exovar=exovar, is_equation=True)


test_tree = BTreeStoreOperations.multiply(bt_S,bt_T)






'''
5) En cualquier árbol que se construya, se debe cumplir la condición de que si todos
los valores que están por debajo de un nodo son iguales, el árbol se poda (o no se sigue construyendo).
'''


'''–
6) Estudiar cómo habría que ordenar las operaciones para calcular P_{t+1}(U) en EMCC, 
considerando que se utilizan árboles.

 - Hacer alguna prueba rápida que compruebe la eficiencia
 - Implementar la clase de EMCC que haga precomputed y con árboles.

'''


'''
7) Posibles tipos de poda, para árboles que no sean ecuaciones.

    - Podar valores cercanos a 0 reemplazándolos por 0. Esto tendía sentido para P_{t+1}(U).
    Puede que hacerlo en cada iteración sea muy costoso... sólo en la última? Estudiar si tendría
    sentido normalizar tras la poda. Una justificación sería que al podar los valores cercanos a
    0 estaríamos moviéndonos a los puntos extremos de conjunto credal.
    
    - Poda de valores similares según KL (esta es la que siempre hemos utilizado en los artículos).
    Esta tendría sentido utilizarla en los factores que se precomputan (que no sean SEs). Permitiría aproximar EMCC

'''

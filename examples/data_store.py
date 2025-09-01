from bcause.factors import MultinomialFactor
from bcause.factors.values.btreestore import BTreeNode, BTreeStore, BTreeNodeConsecutive

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
var_domain = ["u1", "u2", "u3"]

# different ways of building a node
n1 = BTreeNode.build(variable, var_domain, 0.2, 0.4, left_states=["u1"])
n2 = BTreeNode.build(variable, var_domain, 0.2, 0.4, right_states=["u2", "u3"])
n3 = BTreeNode.build(variable, var_domain, 0.4, 0.3, end_left_exclusive=1)


# a non terminal node
nested_nodes = BTreeNode.build("X", ["x1", "x2"], 0.33, n1)

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


n = BTreeNodeNonConsecutive(variable = "U", var_domain=var_domain, left_child=0.5, right_child=0.25, left_states=["u2"])
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


'''
3) Implementar BTreeStore._build_from_equation(table, exovar)

    - Similar a BTreeStore._build_from_table construye un árbol a partir de una ecuación. 
    - El árbol siempre tendrá la variable exógena en la base del árbol.
    - Los nodos de la variable exógena serán de tipo BTreeNodeNonConsecutive
    - Los nodos de las variables endógenas serán de tipo BTreeNodeConsecutive.
    - Las hojas siempre serán 0 o 1
    - Después de implementar esto, intentar ver si se ahorra espacio.
'''


'''
4) Adaptar las operaciones en BTreeStoreOperations para que funcionen con nodos de tipo BTreeNodeNonConsecutive.

    - En las operaciones binarias se mantiene el tipo de nodo del primer operando.
    - En las operaciones binarias no se puede dar el caso de que se opere con dos tipos distintos

'''


'''
5) En cualquier árbol que se construya, se debe cumplir la condición de que si todos
los valores que están por debajo de un nodo son iguales, el árbol se poda (o no se sigue construyendo).
'''


'''
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

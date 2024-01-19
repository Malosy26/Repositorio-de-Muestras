Primero gracias por su interes y ahora vamos a ello.

El repositorio esta realizado en 4 notebook cuyo contenido expreso brevemente aqui:

1-EjercicioClasificacion-> Primera toma de contacto con el ejercicio lo mas importante de aqui es 
luego de probar el modelo la visualizacion de las features importances ya que esto tendra peso en las 
decisiones de los siguiente notebook y la prueba de h2o.

2-EjercicioClasificacionCodigo->Este notebook se dedica exclusivamente a probar los diferentes modelos de clasificación
para ver con cual nos acercamos mas a una mayor presición.Crea una funcion por cada modelo.

AQUI EMPIEZAN LAS VERDADERAS PRUEBAS

3->CreditoLimpio->
Se prepara y presenta el modelo en de 2 formas sin los nan y con los nan tratados.
Para el trato de los nan se realizan 2 tecnicas 1 a mano para calcular la media de los vecinos y
Knnimputer.
Se mapean las categoricas.
Se realiza y ajusta el modelo.

4->CreditoLimpio2->
Para no tener demasiado codigo y que sea legible se planteo hacer el mismo ejercicio que el anterior
solo que en esta ocasion en lugar de mapear las categoricas se utilizo un dummies. Las metricas
resultantes fueron peores que la ocasion anterior.

La libreria de uso propio y para este ejercicio esta realizada para poder usar las funciones sin tener que tenerlas en el notebook


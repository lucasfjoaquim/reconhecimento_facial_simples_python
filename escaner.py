from deepface import DeepFace

# Define the paths to the two known images and the unknown image
image_pessoa1 = "images/pessoa1.jpg"
image_pessoa2 = "images/pessoa2.jpg"
image_desconhecida = "images/terceira_pessoa.jpg"

# Perform face recognition using DeepFace
resultados = DeepFace.verify(image_pessoa1, image_desconhecida, model_name='Facenet')
is_pessoa1 = resultados["verified"]

if not is_pessoa1:
    resultados = DeepFace.verify(image_pessoa2, image_desconhecida, model_name='Facenet')
    is_pessoa2 = resultados["verified"]
else:
    is_pessoa2 = False

# Check the results and print the corresponding message
if is_pessoa1:
    print("A terceira foto pertence à pessoa 1.")
elif is_pessoa2:
    print("A terceira foto pertence à pessoa 2.")
else:
    print("A terceira foto não pertence a nenhuma das duas pessoas.")
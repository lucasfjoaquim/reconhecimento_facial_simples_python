from deepface import DeepFace

# Define o caminho das duas imagens de base (pessoa1.jpg e pessoa2.jpg) e a imagen desconhecida que pode pertencer a uma das pessoas.
image_pessoa1 = "images/pessoa1.jpg"
image_pessoa2 = "images/pessoa2.jpg"
image_desconhecida = "images/terceira_pessoa.jpg"

# realiza o reconhecimento facial com deepface
resultados = DeepFace.verify(image_pessoa1, image_desconhecida, model_name='Facenet')
is_pessoa1 = resultados["verified"]

if not is_pessoa1:
    resultados = DeepFace.verify(image_pessoa2, image_desconhecida, model_name='Facenet')
    is_pessoa2 = resultados["verified"]
else:
    is_pessoa2 = False

# analisa os resultados
if is_pessoa1:
    print("A terceira foto pertence à pessoa 1.")
elif is_pessoa2:
    print("A terceira foto pertence à pessoa 2.")
else:
    print("A terceira foto não pertence a nenhuma das duas pessoas.")
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app, origins='http://localhost:5173')

model = load_model('modelo.h5')
classes = ['arromantica', 'asexual', 'bear', 'bisexual', 'demisexual', 'fluido', 'gay', 'intersexual', 'lesbica', 'nobinario', 'pansexual', 'queer', 'trans']

history_texts = {
    'arromantica': {
        'text': 'La comunidad arromántica comprende personas que experimentan poca o ninguna atracción romántica, aunque pueden tener relaciones profundas y significativas sin romanticismo. Esta orientación incluye un espectro de experiencias, como grisromántico y demiromántico. Las personas arrománticas pueden enfrentar estigmatización debido a las expectativas sociales de las relaciones románticas.'
    },
    'asexual': {
        'text': 'La comunidad asexual comprende personas que experimentan poca o ninguna atracción sexual hacia otros. Esto no implica que no puedan tener relaciones emocionales o románticas, sino que la atracción sexual no es un factor para ellos. La asexualidad es una orientación sexual en sí misma y puede coexistir con cualquier orientación romántica, como arromántica, biromántica o heterorromántica.'
    },
    'bear': {
        'text': 'En la comunidad LGBTQ+, el término bear (oso en inglés) se refiere a un subgrupo de hombres gay, bisexuales y otros hombres que exhiben características como una apariencia robusta, con vello facial y corporal, y una personalidad afable y amigable. Los osos son una parte importante de la diversidad dentro de la comunidad LGBTQ+ y tienen su propia cultura y eventos.'
    },
    'bisexual': {
        'text': 'La bisexualidad es una orientación sexual donde una persona siente atracción hacia más de un género. Esta atracción puede ser romántica, sexual o emocional y no necesariamente en igual medida hacia todos los géneros. La bisexualidad es una identidad válida y significativa dentro de la comunidad LGBTQ+. A menudo enfrenta estigmatización y malentendidos, pero también es una parte integral de la diversidad sexual.'
    },
    'demisexual': {
        'text': 'La demisexualidad es una orientación sexual en la que una persona solo siente atracción sexual después de haber formado un fuerte vínculo emocional con alguien. Esta conexión emocional es esencial para que la atracción sexual se desarrolle. Los demisexuales pueden identificarse con cualquier orientación romántica y sexual, y su experiencia de atracción sexual es distinta de la atracción inmediata o superficial. La demisexualidad es una parte del espectro asexual.'
    },
    'fluido': {
        'text': '"Fluido" describe identidades que cambian con el tiempo. En orientación sexual fluida, una persona experimenta atracción que varía hacia diferentes géneros. En género fluido, la identidad de género de una persona cambia o fluctúa entre diferentes expresiones. Ambas experiencias subrayan la flexibilidad y diversidad en la forma en que las personas pueden vivir su sexualidad y género.'
    },
    'gay': {
        'text': 'Ser gay significa sentirse atraído romántica o sexualmente hacia personas del mismo género. Esta identidad se refiere principalmente a hombres que se sienten atraídos por otros hombres. La homosexualidad es una orientación sexual válida y forma parte integral de la comunidad LGBTQ+. Ser gay implica una identidad y una experiencia que, al igual que cualquier otra orientación, merece respeto y reconocimiento.'
    },
    'intersexual': {
        'text': 'La intersexualidad se refiere a una condición en la que una persona nace con características sexuales que no encajan en las categorías típicas de masculino o femenino. Esto puede incluir variaciones en los genitales, cromosomas, o características sexuales secundarias. Las personas intersexuales pueden tener una identidad de género masculina, femenina o no binaria.'
    },
    'lesbica': {
        'text': 'Ser lesbiana significa que una mujer siente atracción romántica y/o sexual hacia otras mujeres. Esta identidad es una orientación sexual dentro de la comunidad LGBTQ+ y se basa en la experiencia de atracción entre mujeres. '
    },
    'nobinario': {
        'text': 'El término "no binario" se refiere a personas cuya identidad de género no encaja en las categorías tradicionales de masculino o femenino. Las personas no binarias pueden experimentar una identidad de género que es una mezcla de ambos géneros, ninguno de ellos, o que cambia con el tiempo.'
    },
    'pansexual': {
        'text': 'La pansexualidad es una orientación sexual en la que una persona siente atracción romántica o sexual hacia otros independientemente de su género. Esto significa que el género de la persona con la que se relacionan no es un factor determinante en su atracción. '
    },
    'queer': {
        'text': '"Queer" es un término amplio y flexible que se utiliza para describir identidades de género y orientaciones sexuales que no se ajustan a las normas tradicionales de heterosexualidad y binarismo de género. A menudo se usa para abarcar una variedad de identidades y experiencias dentro de la comunidad LGBTQ+'
    },
    'trans': {
        'text': '"Trans" es una abreviatura de "transgénero", que se refiere a personas cuya identidad de género difiere del sexo asignado al nacer. Los trans pueden identificarse como hombres, mujeres, no binarios, o cualquier otra identidad de género. La transición puede involucrar cambios en la apariencia, el nombre y, a veces, tratamientos médicos.'
    }
}



def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_string

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.files['image']
    img = cv2.imdecode(np.fromstring(image_data.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = np.array(img).reshape(-1, 64, 64, 1)
    
    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]
    
    response_data = {'predicted_class': predicted_class}
    
    if predicted_class in history_texts:
        response_data['history_text'] = history_texts[predicted_class]['text']
        # response_data['image'] = encode_image(history_texts[predicted_class]['image_path'])
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

from chatbot.data import training_data
from chatbot.models import buid_and_train_model, load_model, predict_answer
def main():
    model, vectorizer, unique_answer = load_model()
    if model is None:
        model, vectorizer, unique_answer = buid_and_train_model(training_data)  
        print("\n Chatbot listo, Escribe " 'salir' "para terminar. \n")
    while True:
        user = input("Tu: ").strip()
        if user.lower() in {"salir","exit","quit"}:
            print("Bot: !Hasta pronto¡")
            break
        response = predict_answer(model, vectorizer, unique_answer, user)
        print("Bot:", response)
if __name__ == "__main__":
    main() 
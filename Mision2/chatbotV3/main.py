from chatbot.data import training_data 
from chatbot.model import build_and_train_model, predict_answer, load_model 

def chat(model,vectorizer, unique_answer):
    """Inicia modelo de conversación"""
    print("\n 💬 Chat iniciado. Escriba para salir terminar \n")
    while True:
        user = input("Tú: ").strip()
        if user.lower() in {"salir", "exit", "quit"}:
            print("Bot: ¡Hasta pronto!")
            break
        response = predict_answer(model, vectorizer, unique_answer, user)
        print("Bot:", response)
    
    
    
        
def main():
    #Intentar cargar modelo
    model, vectorizer, unique_answer = load_model()
    #Menu principal
    while True:
        print("\n===🤖 MENÚ PRINCIPAL DEL CHATBOT")
        print("1️⃣ chatea con el modelo")
        print("2️⃣ Reentrena el modelo")
        print("3️⃣ Salir")
        opcion=input("\n Selecciona una opción (1-3): ").strip()
        if opcion == "1":
            if model is None:
                print("\n ⚠️ No hay modelo entrenado. Entrena el modelo primero")
            else:
                chat(model, vectorizer, unique_answer)
                
        elif opcion == "2":
            print("\n 🔄 Reentrenando el modelo con los nuevos datos...")
            model, vectorizer, unique_answer = build_and_train_model(training_data)
            print("🆗 Modelo actualizado correctamente")
        elif opcion == "3":
            print("\n 👌 ¡Hasta luego!")
            break
        else:
            print("\n ❌ Opción no válida. Intenta nuevamente.")
            
     
if __name__ == "__main__":
    main()
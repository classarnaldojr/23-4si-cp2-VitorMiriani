import cv2
import mediapipe as mp

# Desenha o traçados nos videos
mp_drawing = mp.solutions.drawing_utils
# estilos para desenhar os traços
mp_drawing_styles = mp.solutions.drawing_styles
# detecta mãos em imagens e videos
mp_hands = mp.solutions.hands


# identifica o gesto da mão
def getHand(hand_landmarks):
    marcas = []
    for marca in hand_landmarks.landmark:
        marcas.append((marca.x, marca.y, marca.z))

    # Calcula a distância entre os dedos 
    dist1 = ((marcas[8][0] - marcas[12][0])**2 +
             (marcas[8][1] - marcas[12][1])**2)**0.5
    dist2 = ((marcas[8][0] - marcas[4][0])**2 +
             (marcas[8][1] - marcas[4][1])**2)**0.5

    # Verifica se o gesto corresponde a pedra, papel ou tesoura
    if dist1 < 0.04 and dist2 < 0.04:
        return "Pedra"
    elif dist1 > 0.06 and dist2 > 0.06:
        return "Tesoura"
    else:
        return "Papel"


# captura o video
cap = cv2.VideoCapture('pedra-papel-tesoura.mp4')

# configuracoes para identificar as maos
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    first_player_gesture = None
    second_player_gesture = None
    winning_player = None  # número do jogador que venceu o round
    scores = [0, 0]

    while True:
        success, img = cap.read()

        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)

        # identificando as maos e desenhando os pontos
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        hls = results.multi_hand_landmarks

        # Quando duas maos forem encontradas
        if hls and len(hls) == 2:
            # menor valor de X para a primeira mao detectada
            min_x_hand_1 = min(list(
                map(lambda l: l.x, hls[0].landmark)))
            # menor valor de X para a segunda mao detectada
            min_x_hand_2 = min(list(
                map(lambda l: l.x, hls[1].landmark)))
            first_player_hand = hls[0] if min_x_hand_1 < min_x_hand_2 else hls[1]
            second_player_hand = hls[0] if min_x_hand_1 > min_x_hand_2 else hls[1]

            if (getHand(second_player_hand) != second_player_gesture or getHand(first_player_hand) != first_player_gesture):

                print("Primeira mão", min_x_hand_1)
                print("Segunda mão", min_x_hand_2)
                # pega o gesto da mao da direita
                second_player_gesture = getHand(second_player_hand)

                # pega o gesto da mao da esquerda
                first_player_gesture = getHand(first_player_hand)

                # condicionais para definir o vencedor
                if success:
                    if first_player_gesture == second_player_gesture:
                        winning_player = 0
                    elif first_player_gesture == "Papel" and second_player_gesture == "Pedra":
                        winning_player = 1
                    elif first_player_gesture == "Papel" and second_player_gesture == "Tesoura":
                        winning_player = 2
                    elif first_player_gesture == "Pedra" and second_player_gesture == "Tesoura":
                        winning_player = 1
                    elif first_player_gesture == "Pedra" and second_player_gesture == "Papel":
                        winning_player = 2
                    elif first_player_gesture == "Tesoura" and second_player_gesture == "Papel":
                        winning_player = 1
                    elif first_player_gesture == "Tesoura" and second_player_gesture == "Pedra":
                        winning_player = 2
                    else:
                        print("Not identified")
                else:
                    success = False

                if winning_player == 1:
                    scores[0] += 1
                elif winning_player == 2:
                    scores[1] += 1

        round_result = "Empate" if winning_player == 0 else f"Jogador {winning_player} Ganhou!"
        # adiciona o texto na tela.
        cv2.putText(img, round_result, (600, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        cv2.putText(img, str("Jogador 1"), (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        cv2.putText(img, first_player_gesture, (100, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        cv2.putText(img, str(scores[0]), (100, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        cv2.putText(img, str('Jogador 2'), (1400, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        cv2.putText(img, second_player_gesture, (1400, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        cv2.putText(img, str(scores[1]), (1400, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)

        # Cria uma janela e define o tamanho
        cv2.namedWindow('PPT', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('PPT', 1920, 1080)
        cv2.imshow('PPT', img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

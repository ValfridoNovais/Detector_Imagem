import cv2
import sys
sys.path.append("C:/Repositorios_GitHube/MeusProjetos/Detector_Imagem/miniatura/Lib/site-packages")
import face_recognition
import pandas as pd
from datetime import datetime
import openpyxl

# Carregar arquivo XLS
arquivo_xls = "C:/Repositorios_GitHube/MeusProjetos/Detector_Imagem/xls/dados_acesso.xlsx"
#arquivo_xls = 'dados_acesso.xlsx'
planilha_permitidos = 'PERMITIDOS'
planilha_registro = 'ENTRADA'

# Função para carregar as imagens conhecidas e seus dados
def carregar_dados_autorizados():
    df = pd.read_excel(arquivo_xls, sheet_name=planilha_permitidos)
    rostos_conhecidos = []
    dados_conhecidos = []

    for index, row in df.iterrows():
        # Carregar a imagem da pessoa autorizada
        imagem = face_recognition.load_image_file(row['Caminho da Foto'])
        encoding = face_recognition.face_encodings(imagem)[0]  # Codificação facial

        rostos_conhecidos.append(encoding)
        dados_conhecidos.append((row['Nome'], row['NR PM']))

    return rostos_conhecidos, dados_conhecidos

# Função para registrar a entrada
def registrar_entrada(nome, nr_pm):
    df = pd.read_excel(arquivo_xls, sheet_name=planilha_registro)
    data_hora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Criar nova linha de registro
    novo_registro = {'Nome': nome, 'NR PM': nr_pm, 'Data': data_hora.split()[0], 'Hora': data_hora.split()[1]}
    df = df.append(novo_registro, ignore_index=True)

    # Salvar registro no arquivo
    with pd.ExcelWriter(arquivo_xls, mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=planilha_registro, index=False)

# Função principal para captura de vídeo e reconhecimento facial
def reconhecer_rostos():
    rostos_conhecidos, dados_conhecidos = carregar_dados_autorizados()

    # Inicializar captura de vídeo (ajustar o índice da câmera conforme necessário)
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capturar frame da câmera
        ret, frame = video_capture.read()

        # Converter a imagem capturada de BGR (OpenCV) para RGB (face_recognition)
        rgb_frame = frame[:, :, ::-1]

        # Localizar rostos no frame
        localizacoes_rostos = face_recognition.face_locations(rgb_frame)
        codificacoes_rostos = face_recognition.face_encodings(rgb_frame, localizacoes_rostos)

        # Para cada rosto detectado, tentar reconhecer
        for (top, right, bottom, left), face_encoding in zip(localizacoes_rostos, codificacoes_rostos):
            # Comparar o rosto capturado com os rostos conhecidos
            matches = face_recognition.compare_faces(rostos_conhecidos, face_encoding)
            nome, nr_pm = "Desconhecido", None

            # Verificar se encontramos uma correspondência
            if True in matches:
                match_index = matches.index(True)
                nome, nr_pm = dados_conhecidos[match_index]

                # Registrar a tentativa de entrada
                registrar_entrada(nome, nr_pm)
                print(f"Acesso liberado para {nome} (NR PM: {nr_pm})")
            else:
                print("Acesso negado")

            # Desenhar um retângulo ao redor do rosto
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Mostrar o nome da pessoa no retângulo
            cv2.putText(frame, nome, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Exibir o frame processado
        cv2.imshow('Reconhecimento Facial', frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar captura de vídeo e fechar janelas
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reconhecer_rostos()

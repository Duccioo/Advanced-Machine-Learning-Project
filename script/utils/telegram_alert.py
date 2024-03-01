import time
import requests
import urllib
import os
from dotenv import load_dotenv

load_dotenv()
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]


def alert(string, path, run_step=0, total_step=0, testo_aggiuntivo=""):
    # send telegram message

    # prendo il path di dove sta il checkpoint e mi ricavo il FOLD corrispondente nome
    project_name = path.split(os.sep)[-3]

    if string == "TRAINING":

        if run_step == 0:
            # hai appena iniziato
            testo = (
                "Grande " + " hai avviato la fase di TEST!" + "\n " + testo_aggiuntivo
            )
        elif (run_step + 1) / total_step >= 1:
            # finito
            testo = "Non ci credo...\n" + "HAI FINITO"
        else:
            testo = (
                f"TRAINING {str(round((run_step / total_step) * 100, 2))}% : <b>#{project_name}</b> \n"
                + str(testo_aggiuntivo)
            )

        send_photo(CHAT_ID, path, caption=testo)
        time.sleep(10)


def send_photo(chat_id, img_path, caption=""):
    caption_parse = caption
    method = "sendPhoto"
    params = {"chat_id": chat_id, "caption": caption_parse, "parse_mode": "HTML"}
    files = {"photo": open(img_path, "rb")}
    resp = requests.post(
        "https://api.telegram.org/bot{}/".format(TELEGRAM_TOKEN) + method,
        params,
        files=files,
    )
    return resp


def send_message(chat_id, testo):
    tot = urllib.parse.quote_plus(testo)
    url = "https://api.telegram.org/bot{}/".format(
        TELEGRAM_TOKEN
    ) + "sendMessage?text={}&chat_id={}".format(tot, chat_id)
    get_url(url)


def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


if __name__ == "__main__":
    pass

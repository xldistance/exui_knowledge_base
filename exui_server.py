import sys, os, json, argparse
from threading import Timer, Lock

from flask import Flask, render_template, request
from flask import Response, stream_with_context
from waitress import serve
import webbrowser
import torch

from backend.models import update_model, load_models, get_model_info, list_models, remove_model, load_model, unload_model, get_loaded_model
from backend.config import set_config_dir, global_state
from backend.sessions import list_sessions, set_session, get_session, get_default_session_settings, new_session, delete_session, set_cancel_signal
from backend.notepads import list_notepads, set_notepad, get_notepad, get_default_notepad_settings, new_notepad, delete_notepad, set_notepad_cancel_signal
from backend.prompts import list_prompt_formats
from backend.settings import get_settings, set_settings

if os.name == "nt":
    # Fix Windows inferring text/plain MIME type for static files
    # https://stackoverflow.com/questions/59355194/
    import mimetypes
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

app = Flask("ExUI")
app.static_folder = 'static'
api_lock = Lock()
api_lock_cancel = Lock()

parser = argparse.ArgumentParser(description="ExUI, chatbot UI for ExLlamaV2")
parser.add_argument("-host", "--host", type = str, help = "IP:PORT eg, 0.0.0.0:5000", default = "localhost:5000")
parser.add_argument("-d", "--dir", type = str, help = "Location for user data and sessions, default: exui_dialogue", default = "exui_dialogue")
parser.add_argument("-v", "--verbose", action = "store_true", help = "Verbose (debug) mode")
parser.add_argument("-nb,", "--no_browser", action = "store_true", help = "Don't launch browser on startup")
args = parser.parse_args()

verbose = args.verbose
no_browser = args.no_browser

@app.route("/")
def home():
    # global api_lock, verbose
    if verbose: print("/")
    # with api_lock:
    return render_template("index.html")

@app.route("/api/list_models")
def api_list_models():
    global api_lock, verbose
    if verbose: print("/api/list_models")
    with api_lock:
        m, c = list_models()
        result = { "result": "ok",
                   "models": m,
                   "current_model": c }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/get_model_info", methods=['POST'])
def api_get_model_info():
    global api_lock, verbose
    if verbose: print("/api/get_model_info")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        info = get_model_info(data)
        if info: result = { "result": "ok",
                            "model_info": info }
        else: result = { "result": "fail" }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/update_model", methods=['POST'])
def api_update_model():
    global api_lock, verbose
    if verbose: print("/api/update_model")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        i = update_model(data["model_info"])
        result = { "result": "ok", "new_model_uuid": i }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/load_model", methods=['POST'])
def api_load_model():
    global api_lock, verbose
    if verbose: print("/api/load_model")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        if verbose: print("-> ...")
        result = Response(stream_with_context(load_model(data)), mimetype = 'application/json')
        if verbose: print("->", result)
        return result

@app.route("/api/unload_model")
def api_unload_model():
    global api_lock, verbose
    if verbose: print("/api/unload_model")
    with api_lock:
        result = unload_model()
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/list_sessions")
def api_list_sessions():
    global api_lock, verbose
    if verbose: print("/api/list_sessions")
    with api_lock:
        s, c = list_sessions()
        result = { "result": "ok", "sessions": s, "current_session": c }
        if verbose: print("-> (...)")
        return json.dumps(result) + "\n"

@app.route("/api/get_default_settings")
def api_get_default_settings():
    global api_lock, verbose
    if verbose: print("/api/get_default_settings")
    with api_lock:
        result = { "result": "ok",
                   "session_settings": get_default_session_settings(),
                   "notepad_settings": get_default_notepad_settings(),
                   "prompt_formats": list_prompt_formats() }
        return json.dumps(result) + "\n"

@app.route("/api/set_session", methods=['POST'])
def api_set_session():
    global api_lock, verbose
    if verbose: print("/api/set_session")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        session = set_session(data)
        if session is not None:
            result = { "result": "ok",
                       "session": session,
                       "prompt_formats": list_prompt_formats() }
            if verbose: print("-> (...)")
        else:
            result = { "result": "fail" }
            if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/new_session", methods=['POST'])
def api_new_session():
    global api_lock, verbose
    if verbose: print("/api/new_session")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        session = new_session()
        current_session = get_session()
        if "settings" in data: current_session.update_settings(data["settings"])
        if "user_input_text" in data: current_session.user_input(data)
        if "new_name" in data: current_session.rename(data)
        result = { "result": "ok", "session": session }
        if verbose: print("-> (...)")
        return json.dumps(result) + "\n"

@app.route("/api/rename_session", methods=['POST'])
def api_rename_session():
    global api_lock, verbose
    if verbose: print("/api/rename_session")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        s = get_session()
        s.rename(data)
        result = { "result": "ok" }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/update_settings", methods=['POST'])
def api_update_settings():
    global api_lock, verbose
    if verbose: print("/api/update_settings")
    with api_lock:
        s = get_session()
        data = request.get_json()
        if verbose: print("<-", data)
        s.update_settings(data["settings"])
        result = { "result": "ok" }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/user_input", methods=['POST'])
def api_user_input():
    global api_lock, verbose
    if verbose: print("/api/user_input")
    with api_lock:
        s = get_session()
        data = request.get_json()
        if verbose: print("<-", data)
        new_block = s.user_input(data)
        result = { "result": "ok", "new_block": new_block }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/list_prompt_formats")
def api_list_prompt_formats():
    global api_lock, verbose
    if verbose: print("/api/list_prompt_formats")
    with api_lock:
        result = {"result": "ok", "prompt_formats": list_prompt_formats()}
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/delete_block", methods=['POST'])
def api_delete_block():
    global api_lock, verbose
    if verbose: print("/api/delete_block")
    with api_lock:
        s = get_session()
        data = request.get_json()
        if verbose: print("<-", data)
        s.delete_block(data["block_uuid"])
        result = { "result": "ok" }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/edit_block", methods=['POST'])
def api_edit_block():
    global api_lock, verbose
    if verbose: print("/api/edit_block")
    with api_lock:
        s = get_session()
        data = request.get_json()
        if verbose: print("<-", data)
        s.edit_block(data["block"])
        result = { "result": "ok" }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/generate", methods=['POST'])
def api_generate():
    global api_lock, verbose
    if verbose: print("/api/generate")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        s = get_session()
        if verbose: print("-> ...");
        result = Response(stream_with_context(s.generate(data)), mimetype = 'application/json')
        if verbose: print("->", result)
        return result

@app.route("/api/cancel_generate")
def api_cancel_generate():
    global api_lock_cancel, verbose
    if verbose: print("/api/cancel_generate")
    with api_lock_cancel:
        set_cancel_signal()
        result = { "result": "ok" }
        if verbose: print("->", result)
        return result

@app.route("/api/delete_session", methods=['POST'])
def api_delete_session():
    global api_lock, verbose
    if verbose: print("/api/delete_session")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        delete_session(data["session_uuid"]);
        result = { "result": "ok" }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/remove_model", methods=['POST'])
def api_remove_model():
    global api_lock, verbose
    if verbose: print("/api/remove_model")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        remove_model(data)
        result = { "result": "ok" }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/get_settings")
def api_get_settings():
    global api_lock, verbose
    if verbose: print("/api/get_settings")
    with api_lock:
        settings = get_settings()
        result = { "result": "ok",
                   "settings": settings }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/set_settings", methods=['POST'])
def api_set_settings():
    global api_lock, verbose
    if verbose: print("/api/set_settings")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        set_settings(data["settings"])
        result = { "result": "ok" }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/list_notepads")
def api_list_notepads():
    global api_lock, verbose
    if verbose: print("/api/list_notepads")
    with api_lock:
        n, c = list_notepads()
        result = { "result": "ok", "notepads": n, "current_notepad": c }
        if verbose: print("-> (...)")
        return json.dumps(result) + "\n"

@app.route("/api/set_notepad", methods=['POST'])
def api_set_notepad():
    global api_lock, verbose
    if verbose: print("/api/set_notepad")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        r = set_notepad(data)
        if r["notepad"] is not None:
            result = { "result": "ok",
                       "notepad": r["notepad"] }
            if "tokenized_text" in r:
                result["tokenized_text"] = r["tokenized_text"]
            if verbose: print("-> (...)")
        else:
            result = { "result": "fail" }
            if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/new_notepad", methods=['POST'])
def api_new_notepad():
    global api_lock, verbose
    if verbose: print("/api/new_notepad")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        notepad = new_notepad()
        if "settings" in data: get_notepad().update_settings(data["settings"])
        if "text" in data: get_notepad().set_text(data["text"])
        if "new_name" in data: get_notepad().rename(data)
        result = { "result": "ok", "notepad": notepad }
        if verbose: print("-> (...)")
        return json.dumps(result) + "\n"

@app.route("/api/rename_notepad", methods=['POST'])
def api_rename_notepad():
    global api_lock, verbose
    if verbose: print("/api/rename_notepad")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        s = get_notepad()
        s.rename(data)
        result = { "result": "ok" }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/delete_notepad", methods=['POST'])
def api_delete_notepad():
    global api_lock, verbose
    if verbose: print("/api/delete_notepad")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        delete_notepad(data["notepad_uuid"]);
        result = { "result": "ok" }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/update_notepad_settings", methods=['POST'])
def api_update_notepad_settings():
    global api_lock, verbose
    if verbose: print("/api/update_notepad_settings")
    with api_lock:
        n = get_notepad()
        data = request.get_json()
        if verbose: print("<-", data)
        n.update_settings(data["settings"])
        result = { "result": "ok" }
        if verbose: print("->", result)
        return json.dumps(result) + "\n"

@app.route("/api/set_notepad_text", methods=['POST'])
def api_set_notepad_text():
    global api_lock, verbose
    if verbose: print("/api/set_notepad_text")
    with api_lock:
        n = get_notepad()
        data = request.get_json()
        if verbose: print("<-", data)
        n.set_text(data["text"])
        tokenized_text = n.get_tokenized_text()
        result = { "result": "ok", "tokenized_text": tokenized_text }
        if verbose: print("-> (...)")
        return json.dumps(result) + "\n"

@app.route("/api/notepad_single_token", methods=['POST'])
def api_notepad_single_token():
    global api_lock, verbose
    if verbose: print("/api/notepad_single_token")
    with api_lock:
        n = get_notepad()
        data = request.get_json()
        if verbose: print("<-", data)
        result = n.generate_single_token(data)
        if verbose: print("-> (...)")
        return json.dumps(result) + "\n"


@app.route("/api/notepad_generate", methods=['POST'])
def api_notepad_generate():
    global api_lock, verbose
    if verbose: print("/api/notepad_generate")
    with api_lock:
        data = request.get_json()
        if verbose: print("<-", data)
        n = get_notepad()
        if verbose: print("-> ...");
        result = Response(stream_with_context(n.generate(data)), mimetype = 'application/json')
        if verbose: print("->", result)
        return result

@app.route("/api/cancel_notepad_generate")
def api_cancel_notepad_generate():
    global api_lock_cancel, verbose
    if verbose: print("/api/cancel_notepad_generate")
    with api_lock_cancel:
        set_notepad_cancel_signal()
        result = { "result": "ok" }
        if verbose: print("->", result)
        return result


# Prepare torch

# torch.cuda._lazy_init()

# Prepare config
dir_path = set_config_dir(args.dir)
print(f" -- User dir: {dir_path}")
global_state.load()
load_models()

# Start server

machine = args.host
host, port = machine.split(":")
browser_start = False

if host == "localhost" and not no_browser:
    Timer(1, lambda: webbrowser.open(f'http://{machine}/')).start()
    browser_start = True

print(f" -- Starting server on {host} port {port}")
if browser_start:
    print(f" -- Opening UI in default web browser")

serve(app, host = host, port = port, threads = 8)

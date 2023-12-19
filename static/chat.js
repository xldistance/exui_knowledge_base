
import * as util from "./util.js";
import * as mainmenu from "./mainmenu.js";
import * as globals from "./globals.js";
import * as controls from "./controls.js";
import * as overlay from "./overlay.js";
import * as chatsettings from "./chatsettings.js";
import * as roles from "./roles.js";

export class Chat {
    constructor() {
        this.page = util.newDiv(null, "models");

        let layout = util.newVFlex("grow");
        this.page.appendChild(layout);

        let layout_l = util.newHFlex();
        this.sessionList = util.newDiv(null, "session-list");
        this.sessionView = util.newDiv(null, "session-view");
        let panel = util.newDiv(null, "session-list-controls");
        layout.appendChild(layout_l);
        layout_l.appendChild(this.sessionList);
        layout_l.appendChild(panel);
        layout.appendChild(this.sessionView);

        this.removeButton = new controls.LinkButton("✖ Delete session", "✖ Confirm", () => { this.deleteSession(this.lastSessionUUID); });
        panel.appendChild(this.removeButton.element);

        this.lastSessionUUID = null;

        this.items = new Map();
        this.labels = new Map();
        this.currentView = null;
    }

    onEnter(getResponse = false) {
        fetch("/api/list_sessions")
        .then(response => response.json())
        .then(response => {
            globals.receiveGlobals(response);
            this.populateSessionList(response, getResponse);
        });
    }

    populateSessionList(response, getResponse = false) {
        //console.log(response);

        this.sessionList.innerHTML = "";

        for (let session_uuid in response.sessions)
            if (response.sessions.hasOwnProperty(session_uuid))
                this.addSession(response.sessions[session_uuid], session_uuid);

        this.addSession("New session", "new");
        let m = this.lastSessionUUID ? this.lastSessionUUID : "new";
        this.setSession(m, getResponse);
    }

    addSession(name, sessionID) {

        let nd = util.newDiv("session_" + sessionID, "session-list-entry inactive");
        nd.addEventListener("click", () => {
            if (nd.classList.contains("inactive"))
                this.setSession(sessionID);
        });

        let label = new controls.EditableLabel(name, false, (new_name) => {
            this.currentView.setName(new_name, (response) => {
                if (sessionID == "new") {
                    this.lastSessionUUID = response.session.session_uuid;
                    this.onEnter();
                } else {
                }
            });
        });

        let nd2 = util.newIcon(sessionID == "new" ? "session-new-icon" : "session-icon");
        //nd2.classList.add("hidden");

        nd.appendChild(nd2);
        nd.appendChild(label.element);
        this.sessionList.appendChild(nd);
        this.items.set(sessionID, nd);
        this.labels.set(sessionID, label);
    }

    setSession(sessionID, getResponse = false) {
        this.items.forEach((v, k) => {
            let div = this.items.get(k);
            if (k == sessionID) {
                div.classList.add("active");
                div.classList.remove("inactive");
            } else {
                div.classList.remove("active");
                div.classList.add("inactive");
            }
            let label = this.labels.get(k);
            label.setEditable(k == sessionID);
        });

        this.currentView = new SessionView(sessionID, this);
        this.currentView.updateView(getResponse);
        this.sessionView.innerHTML = "";
        this.sessionView.appendChild(this.currentView.element);
        this.lastSessionUUID = sessionID;

        this.removeButton.setEnabled(sessionID && sessionID != "new");
        this.currentView.focusInputField();
    }

    deleteSession(sessionID) {
        let packet = {};
        packet.session_uuid = this.currentView.sessionID;
        util.assert(packet.session_uuid == this.lastSessionUUID);
        fetch("/api/delete_session", { method: "POST", headers: { "Content-Type": "application/json", }, body: JSON.stringify(packet) })
        .then(response => response.json())
        .then(response => {
            this.lastSessionUUID = util.getNextKey(this.items, sessionID);
            this.onEnter();
        });
    }
}

class SessionView {
    constructor(sessionID, parent) {
        this.sessionID = sessionID;
        this.parent = parent;
        this.history = new Map();

        this.element = util.newVFlex();
        this.chatView = util.newDiv("session-view", "session-view");
        this.settingsView = util.newDiv(null, "session-settings");
        this.element.appendChild(this.chatView);
        this.element.appendChild(this.settingsView);

        this.stickyScroll = true;
        this.prevScroll = 0;
        this.chatHistory = util.newDiv("session-view-history", "session-view-history");
        this.chatView.appendChild(this.chatHistory);
        this.chatHistory.addEventListener("scroll", () => {
            let position = this.chatHistory.scrollTop;
            if (this.isNearBottom() && !this.stickyScroll) {
                this.stickyScroll = true;
                //console.log("stick");
            }
            else if (position < this.prevScroll && this.stickyScroll) {
                this.stickyScroll = false;
                //console.log("unstick");
            }
            this.prevScroll = position;
        });

        let surround = util.newDiv(null, "session-input-surround");
        this.chatView.appendChild(surround);

        let sdiv = this.createInputField();
        surround.appendChild(sdiv);
        this.sessionInput = sdiv.children[0];

        this.items = new Map();

    }

    createInputField() {
        let sdiv = util.newVFlex();

        let div = document.createElement("textarea");
        div.className = "session-input";
        div.placeholder = "Type here...";
        div.rows = "1";
        div.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.submitInput();
            }
        });
        div.addEventListener('input', () => { this.inputFieldAutogrow(); });

        this.inputButton = new controls.Button("⏵ Chat", () => { this.submitInput() }, "session-input-button");
        this.cancelButton = new controls.Button("⏹ Stop", () => { this.cancelGen() }, "session-input-button");
        this.inputButton.setHidden(false);
        this.cancelButton.setHidden(true);
        this.inputButton.refresh();
        this.cancelButton.refresh();
        sdiv.appendChild(div);
        sdiv.appendChild(this.inputButton.element);
        sdiv.appendChild(this.cancelButton.element);
        return sdiv;
    }

    focusInputField() {
        this.sessionInput.focus();
    }

    inputFieldAutogrow() {
        this.sessionInput.style.height = 'auto';
        this.sessionInput.style.height = (this.sessionInput.scrollHeight) + 'px';
    }

    updateView(getResponse = false) {
        if (!this.sessionID || this.sessionID == "new") {
            fetch("/api/get_default_settings")
            .then(response => response.json())
            .then(response => {
                globals.receiveGlobals(response);
                this.chatSettings = response.session_settings;
                this.history = new Map();
                this.name = "New session";
                this.populate();
            });
        } else {
            let packet = {};
            packet.session_uuid = this.sessionID;
            fetch("/api/set_session", { method: "POST", headers: { "Content-Type": "application/json", }, body: JSON.stringify(packet) })
            .then(response => response.json())
            .then(response => {
                globals.receiveGlobals(response);
                this.chatSettings = response.session.settings;
                this.history = new Map();
                for (const block of response.session.history)
                    this.history.set(block.block_uuid, block);
                this.name = response.name;
                this.populate();

                if (getResponse && globals.g.loadedModelUUID) this.getModelResponse();
            });
        }
    }

    populate() {
        this.settings = new chatsettings.SessionSettings(this);
        this.settingsView.innerHTML = "";
        this.settingsView.appendChild(this.settings.element);

        this.chatHistory.innerHTML = "";
        for (let block of this.history.values()) this.setChatBlock(block);

        this.scrollToBottom();
        setTimeout(() => {
            this.scrollToBottom();
        }, 200);
    }

    setName(new_name, post = null) {
        let packet = {};
        packet.new_name = new_name;
        if (!this.sessionID || this.sessionID == "new") {
            fetch("/api/new_session", { method: "POST", headers: { "Content-Type": "application/json", }, body: JSON.stringify(packet) })
            .then(response => response.json())
            .then(response => {
                if (post) post(response);
            });
        } else {
            fetch("/api/rename_session", { method: "POST", headers: { "Content-Type": "application/json", }, body: JSON.stringify(packet) })
            .then(response => response.json())
            .then(response => {
                if (post) post(response);
            });
        }
    }

    submitInput() {
        let input = this.sessionInput.value.trim();
        this.sessionInput.value = "";
        this.inputFieldAutogrow();
        this.scrollToBottom();

        if (!this.sessionID || this.sessionID == "new") {
            if (input && input != "") {
                let packet = {};
                packet.user_input_text = input;
                fetch("/api/new_session", { method: "POST", headers: { "Content-Type": "application/json", }, body: JSON.stringify(packet) })
                .then(response => response.json())
                .then(response => {
                    //console.log(response);
                    this.parent.lastSessionUUID = response.session.session_uuid;
                    this.parent.onEnter(true);
                });
            }
        } else {
            if (input && input != "") {
                let packet = {};
                packet.user_input_text = input;
                fetch("/api/user_input", { method: "POST", headers: { "Content-Type": "application/json", }, body: JSON.stringify(packet) })
                .then(response => response.json())
                .then(response => {
                    this.setChatBlock(response.new_block);
                    this.scrollToBottom();
                    if (globals.g.loadedModelUUID) this.getModelResponse();
                });
            } else {
                if (globals.g.loadedModelUUID) this.getModelResponse();
            }
        }
    }

    disableInput() {
        //console.log("disable");
        this.sessionInput.disabled = true;
        this.inputButton.setHidden(true);
        this.cancelButton.setHidden(false);
    }

    enableInput() {
        //console.log("enable");
        this.sessionInput.disabled = false;
        this.inputButton.setHidden(false);
        this.cancelButton.setHidden(true);
    }

    cancelGen() {
        fetch("/api/cancel_generate")
        .then(response => response.json())
        .then(response => {
        });
    }

    isNearBottom() {
        const threshold = 20;
        const position = this.chatHistory.scrollTop + this.chatHistory.offsetHeight;
        const height = this.chatHistory.scrollHeight;
        //console.log(this.chatHistory.scrollTop, this.chatHistory.offsetHeight, this.chatHistory.scrollTop + this.chatHistory.offsetHeight, this.chatHistory.scrollHeight);
        return height - position < threshold;
    }

    scrollToBottom() {
        this.stickyScroll = true;
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                let behavior = globals.g.smoothScrolling ? 'smooth' : 'auto';
                this.chatHistory.scroll({ top: this.chatHistory.scrollHeight, behavior: behavior });
            });
        });
    }

    getModelResponse() {

        this.disableInput();
        let packet = {};

        let timeout = new Promise((resolve, reject) => {
            let id = setTimeout(() => {
                clearTimeout(id);
                reject('No response from server')
            }, 10000)
        });

        let fetchRequest = fetch("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json", },
            body: JSON.stringify(packet)
        });

        const self = this;
        Promise.race([fetchRequest, timeout])
        .then(response => {
            if (response.ok) {
                return response.body;
            } else {
                //appendErrorMessage("Network response was not ok");
                throw new Error("Network response was not ok.");
            }
        })
        .then(stream => {
            let reader = stream.getReader();
            let decoder = new TextDecoder();
            let data = '';
            reader.read().then(function process({done, value}) {
                // console.log("Received chunk:", decoder.decode(value));
                if (done) {
                    //console.log("DONE");
                    self.enableInput();
                    self.focusInputField();
                    return;
                }
                data += decoder.decode(value, {stream: true});
                let lines = data.split('\n');
                for (let i = 0; i < lines.length - 1; i++) {
                    let json = null;
                    try {
                        json = JSON.parse(lines[i]);
                    } catch(e) {
                        console.error("Invalid JSON:", lines[i]);
                        break;
                    }
                    if (json.result == "fail") {
                        console.error('Error:', json.error);
                        self.enableInput();
                        self.focusInputField();
                        return;
                    } else {
                        self.receivedStreamResponse(json);
                    }
                }
                data = lines[lines.length - 1];
                reader.read().then(process);
            });
        })
        .catch(error => {
            console.error('Error:', error);
            self.enableInput();
            self.focusInputField();
        })
    }

    receivedStreamResponse(response) {

        //this.stickyScroll = this.isNearBottom();

        if (response.result == "begin_block") {
            this.currentStreamingBlock = this.setChatBlock(response.block);
        }

        if (response.result == "prompt_eval") {
            this.currentStreamingBlock.setSpinner();
        }

        if (response.result == "stream_to_block") {
            this.currentStreamingBlock.appendText(response.text);
        }

        if (response.result == "ok") {
            this.currentStreamingBlock.set(response.new_block);
        }

        if (this.stickyScroll) this.scrollToBottom();
    }

    setChatBlock(block) {
        let uuid = block.block_uuid;
        if (this.items.has(uuid)) {
            let oldBlock = this.items.get(uuid);
            oldBlock.set(block);
            return oldBlock;
        } else {
            let newBlock = new ChatBlock(this, block);
            this.items.set(uuid, newBlock);
            this.chatHistory.appendChild(newBlock.element);
            return newBlock;
        }
    }

    removeBlock(chatBlock) {
        let uuid = chatBlock.block.block_uuid;
        this.items.delete(uuid);
        this.history.delete(uuid);
        chatBlock.element.remove();
    }
}

class ChatBlock {
    constructor(parent, block) {
        this.parent = parent;
        this.block = block;

        this.element = util.newDiv("chat_block_" + this.block.block_uuid, "session-block");
        this.inner = util.newVFlex();
        this.element.appendChild(this.inner);

        this.avatarImg = util.newDiv(null, "avatar-img");
        this.textBlock = util.newDiv(null, "session-block-text");
        this.inner.appendChild(this.avatarImg);
        this.inner.appendChild(this.textBlock);

        this.updateAvatarImg();
        this.updateText();
        this.updateMeta();

        if (block.author == "user")
            this.element.classList.add("user");

        this.spinnerTimeout = null;
        this.setActions();
    }

    set(block) {
        this.block = block;

        this.updateAvatarImg();
        this.updateText();
        this.updateMeta();
    }

    getRoleID()
    {
        if (!this.parent.chatSettings) return -1;
        let t = this.block.text.toUpperCase();
        for (let i = 0; i < 8; i++)
            if (t.startsWith(this.parent.chatSettings.roles[i].toUpperCase() + ":")) return i;
        return -1;
    }

    updateAvatarImg() {
        let graphic = roles.fallbackAvatar;
        if (this.parent.chatSettings.prompt_format == "Chat-RP")
        {
            let ri = this.getRoleID();
            if (ri != -1) graphic = roles.roleAvatars[ri];
        } else {
            if (this.block.author == "user") graphic = roles.instructAvatars[0];
            if (this.block.author == "assistant") graphic = roles.instructAvatars[1];
        }
        this.avatarImg.innerHTML = "<img src='" + graphic + "' width='64px' draggable='False'>";
    }

    updateText() {
        this.cancelSpinner();
        this.rawtext = this.block.text;

        let name = null;
        let col = roles.fallbackColor;
        let text = this.block.text.trimEnd();

        if (this.parent.chatSettings.prompt_format == "Chat-RP") {
            let ri = this.getRoleID();
            if (ri != -1) {
                col = roles.roleColors[ri];
                name = this.parent.chatSettings.roles[ri];
                text = text.slice(name.length + 1).trimStart();
            }
        } else {
            if (this.block.author == "user") { col = roles.instructColors[0]; name = "User"; }
            if (this.block.author == "assistant") { col = roles.instructColors[1]; name = "Assistant"; }
        }

        let html = "";
        if (name) html += "<div class='name' style='color: " + col + "'>" + name + "</div>"
        html += marked.parse(text);

        this.textBlock.innerHTML = html;
    }

    appendText(new_text) {
        this.cancelSpinner();
        this.block.text += new_text;
        this.updateText();
    }

    cancelSpinner() {
        if (this.spinnerTimeout) {
            clearTimeout(this.spinnerTimeout);
            this.spinnerTimeout = null;
        }
    }

    setSpinner() {
        this.spinnerTimeout = setTimeout(() => {
            this.textBlock.innerHTML += "<div class='spinner'><div></div><div></div><div></div><div></div></div>";
        }, 200);
    }

    updateMeta() {
        if (Object.keys(this.block).includes("meta") && this.block.meta) {
            if (this.block.meta.overflow > 0) {
                let p = document.createElement('p');
                p.classList.add("error");
                p.innerHTML = "‼ Response exceeded " + this.block.meta.overflow + " tokens and was cut short.";
                this.textBlock.appendChild(p);
            }

            if (this.block.meta.canceled) {
                let p = document.createElement('p');
                p.classList.add("error");
                p.innerHTML = "‼ Canceled";
                this.textBlock.appendChild(p);
            }

            let p = document.createElement('p');
            p.classList.add("meta");
            //if (!statsvisible) p.classList.add("hidden");

            let ptps = this.block.meta.prompt_speed.toFixed(2)
            if (this.block.meta.prompt_speed > 50000) ptps = "∞";

            let html = "prompt: " + this.block.meta.prompt_tokens.toFixed(0) + " tokens, " + ptps + " tokens/s";
            html += " ⁄ ";
            html += "response: " + this.block.meta.gen_tokens.toFixed(0) + " tokens, " + this.block.meta.gen_speed.toFixed(2) + " tokens/s";
            p.innerHTML = html;
            this.textBlock.appendChild(p);
        }
    }

    setActions() {
        this.actdiv = util.newDiv(null, "block-actions no-select");

        let span = document.createElement("span");
        span.classList.add("action");
        span.innerHTML = "✕ Delete";
        this.actdiv.appendChild(span);
        span.addEventListener('click', () => {
            this.deleteBlock();
        });

        span = document.createElement("span");
        span.classList.add("action");
        span.innerHTML = "🖉 Edit";
        this.actdiv.appendChild(span);
        span.addEventListener('click', () => {
            this.editBlock()
        });

        this.element.appendChild(this.actdiv);
    }

    deleteBlock() {
        let packet = {};
        packet.block_uuid = this.block.block_uuid;
        fetch("/api/delete_block", { method: "POST", headers: { "Content-Type": "application/json", }, body: JSON.stringify(packet) })
        .then(response => response.json())
        .then(response => {
            this.parent.removeBlock(this);
        });
    }

    editBlock() {
        this.element.classList.add("hidden_h");
        util.makeEditable(this.textBlock, null, (id, val) => {
            //console.log("edit");
            let new_block = { ...this.block };
            new_block.text = val.trim();
            new_block.meta = null;
            let packet = {};
            packet.block = new_block;
            fetch("/api/edit_block", { method: "POST", headers: { "Content-Type": "application/json", }, body: JSON.stringify(packet) })
            .then(response => response.json())
            .then(response => {
                this.block = new_block;
                this.updateText();
                this.updateMeta();
                this.updateAvatarImg();
                this.element.classList.remove("hidden_h");
                this.parent.focusInputField();
            });
        }, (id) => {
            //console.log("not edit");
            this.updateText();
            this.updateMeta();
            this.updateAvatarImg();
            this.element.classList.remove("hidden_h");
            this.parent.focusInputField();
        }, this.rawtext, true);
    }
}


import * as util from "./util.js";

class PageOverlay {
    constructor() {
        this.keyboardDisabled = false;
        document.addEventListener("keydown", (e) => {
            if (this.keyboardDisabled) e.preventDefault();
        });

        this.overlayElement = util.newDiv(null, "page-overlay");
        document.body.appendChild(this.overlayElement);

        this.items = new Map();
    }

    add(mode, control) {
        this.items.set(mode, control);
        this.overlayElement.appendChild(control.element);
        return control;
    }

    setMode(mode = null) {
        if (!mode) {
            this.keyboardDisabled = false;
            this.overlayElement.style.display = "none";
            this.items.forEach((v, k) => { v.setVisible(false); });
        } else {
            this.keyboardDisabled = true;
            this.overlayElement.style.display = "flex";
            this.items.forEach((v, k) => { v.setVisible(mode === k); });
        }
    }
}

class Overlay {
    constructor() {
        this.element = util.newDiv(null, "overlay");
    }

    setVisible(visible) {
        this.element.style.display = visible ? "flex" : "none";
    }
}

class BusyOverlay extends Overlay {
    constructor() {
        super();
        this.element.innerHTML = "<p class='p-header'>Please wait</p>" +
                                 "<div class='spinner'><div></div><div></div><div></div><div></div></div>";
    }
}

class LoadingOverlay extends Overlay {
    constructor() {
        super();
        this.element.innerHTML = "<p class='p-header'>Loading</p>";

        this.box = util.newDiv(null, "progressbar-box");
        this.element.appendChild(this.box);

        this.bar = util.newDiv(null, "progressbar-bar");
        this.box.appendChild(this.bar);
    }

    setProgress(a, b) {
        let percentage = 100 * (a / b);
        this.bar.style.width = percentage + '%';
    }
}

export var pageOverlay = new PageOverlay();
export var busyOverlay = pageOverlay.add("busy", new BusyOverlay());
export var loadingOverlay = pageOverlay.add("loading", new LoadingOverlay());

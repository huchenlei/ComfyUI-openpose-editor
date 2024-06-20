import { app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { ComfyApp } from "../../scripts/app.js";


function addMenuHandler(nodeType, cb) {
    const getOpts = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function () {
        const r = getOpts.apply(this, arguments);
        cb.apply(this, arguments);
        return r;
    };
}

class OpenposeEditorDialog extends ComfyDialog {
    static timeout = 5000;
    static instance = null;

    static getInstance() {
        if (!OpenposeEditorDialog.instance) {
            OpenposeEditorDialog.instance = new OpenposeEditorDialog();
        }

        return OpenposeEditorDialog.instance;
    }

    constructor() {
        super();
        this.element = $el("div.comfy-modal", {
            parent: document.body,
            style: {
                width: "80vw",
                height: "80vh",
            },
        }, [
            $el("div.comfy-modal-content", {
                style: {
                    width: "100%",
                    height: "100%",
                },
            }, this.createButtons()),
        ]);
        this.is_layout_created = false;

        window.addEventListener("message", (event) => {
            if (event.source !== this.iframeElement.contentWindow) {
                return;
            }

            const message = event.data;
            if (message.modalId === 0) {
                const targetNode = ComfyApp.clipspace_return_node;
                const textAreaElement = targetNode.widgets[0].element;
                textAreaElement.value = JSON.stringify(event.data.poses);
                this.close();
            }
        });
    }

    createButtons() {
        const closeBtn = $el("button", {
            type: "button",
            textContent: "Close",
            onclick: () => this.close(),
        });
        return [
            closeBtn,
        ];
    }

    close() {
        super.close();
    }

    async show() {
        if (!this.is_layout_created) {
            this.createLayout();
            this.is_layout_created = true;
            await this.waitIframeReady();
        }

        const targetNode = ComfyApp.clipspace_return_node;
        const textAreaElement = targetNode.widgets[0].element;
        this.element.style.display = "flex";
        this.setCanvasJSONString(textAreaElement.value);
    }

    createLayout() {
        this.iframeElement = $el("iframe", {
            // Change to for local dev
            // src: "http://localhost:5173",
            src: "https://huchenlei.github.io/sd-webui-openpose-editor?theme=dark",
            style: {
                width: "100%",
                height: "100%",
                border: "none",
            },
        });
        const modalContent = this.element.querySelector(".comfy-modal-content");
        while (modalContent.firstChild) {
            modalContent.removeChild(modalContent.firstChild);
        }
        modalContent.appendChild(this.iframeElement);
    }

    waitIframeReady() {
        return new Promise((resolve, reject) => {
            const receiveMessage =  (event) => {
                if (event.source !== this.iframeElement.contentWindow) {
                    return;
                }
                if (event.data.ready) {
                    window.removeEventListener("message", receiveMessage);
                    clearTimeout(timeoutHandle);
                    resolve();
                }
            };
            const timeoutHandle = setTimeout(() => {
                reject(new Error("Timeout"));
            }, OpenposeEditorDialog.timeout);

            window.addEventListener("message", receiveMessage);
        });
    }

    setCanvasJSONString(jsonString) {
        this.iframeElement.contentWindow.postMessage({
            modalId: 0,
            poses: JSON.parse(jsonString)
        }, "*");
    }
}

app.registerExtension({
    name: "huchenlei.EditOpenpose",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "huchenlei.LoadOpenposeJSON") {
            addMenuHandler(nodeType, function (_, options) {
                options.unshift({
                    content: "Open in Openpose Editor",
                    callback: () => {
                        // `this` is the node instance
                        ComfyApp.copyToClipspace(this);
                        ComfyApp.clipspace_return_node = this;

                        const dlg = OpenposeEditorDialog.getInstance();
                        dlg.show();
                    },
                });
            });
        }
    }
});

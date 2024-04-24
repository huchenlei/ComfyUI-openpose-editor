import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { ComfyApp } from "../../scripts/app.js";
import { ClipspaceDialog } from "../../extensions/core/clipspace.js";


(function () {
    const EDITOR_URL = 'https://huchenlei.github.io/sd-webui-openpose-editor/';

    function addMenuHandler(nodeType, cb) {
        const getOpts = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function () {
            const r = getOpts.apply(this, arguments);
            cb.apply(this, arguments);
            return r;
        };
    }

    class OpenPoseEditorDialog extends ComfyDialog {
        static instance = null;

        static getInstance() {
            if (!OpenPoseEditorDialog.instance) {
                OpenPoseEditorDialog.instance = new OpenPoseEditorDialog();
            }

            return OpenPoseEditorDialog.instance;
        }

        constructor() {
            super();
            this.element = $el("div.comfy-modal", {
                id: "comfyui-openpose-editor",
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
                }, [
                    $el("iframe", {
                        src: EDITOR_URL + "?theme=dark",
                        style: {
                            width: "100%",
                            height: "100%",
                        },
                    }),
                    ...this.createButtons(),
                ]),
            ]);
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

        show() {
            this.element.style.display = "flex";
        }
    }

    function isOpenposeEditor(nodeData) {
        return true;
    }

    app.registerExtension({
        name: "huchenlei.OpenPoseEditor",

        init(app) {
            const callback =
                function () {
                    let dlg = OpenPoseEditorDialog.getInstance();
                    dlg.show();
                };

            const context_predicate = () => ComfyApp.clipspace && ComfyApp.clipspace.imgs && ComfyApp.clipspace.imgs.length > 0
            ClipspaceDialog.registerButton("OpenPose Editor", context_predicate, callback);
        },

        async beforeRegisterNodeDef(nodeType, nodeData) {
            // if (!isOpenposeEditor(nodeData)) return;
            if (Array.isArray(nodeData.output) && (nodeData.output.includes("MASK") || nodeData.output.includes("IMAGE"))) {
                addMenuHandler(nodeType, function (_, options) {
                    options.unshift({
                        content: "Open in OpenPose Editor",
                        callback: () => {
                            ComfyApp.copyToClipspace(this);
                            ComfyApp.clipspace_return_node = this;

                            let dlg = OpenPoseEditorDialog.getInstance();
                            dlg.show();
                        },
                    });
                });
            }
        }
    });
})();
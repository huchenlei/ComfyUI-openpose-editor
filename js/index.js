import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { ComfyApp } from "../../scripts/app.js";
import { ClipspaceDialog } from "../../extensions/core/clipspace.js";


(function () {
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
            this.element = $el("div.comfy-modal", { parent: document.body },
                [$el("div.comfy-modal-content",
                    [...this.createButtons()]),
                ]);
        }

        createButtons() {
            return [];
        }

        show() {
            this.mask_image = null;
            self.prompt_points = [];

            this.message_box = $el("p", ["Please wait a moment while the SAM model and the image are being loaded."]);
            this.element.appendChild(this.message_box);

            if (!this.is_layout_created) {
                console.log("Creating layout.");
                // layout
                this.is_layout_created = true;
            }

            // if (ComfyApp.clipspace_return_node) {
            //     this.saveButton.innerText = "Save to node";
            // } else {
            //     this.saveButton.innerText = "Save";
            // }
            // this.saveButton.disabled = true;
            this.element.style.display = "block";
            this.element.style.zIndex = 8888; // NOTE: alert dialog must be high priority.
        }
    }

    // const EDITOR_URL = 'https://huchenlei.github.io/sd-webui-openpose-editor/';
    const EDITOR_URL = 'http://localhost:5173/';
    function createOpenposeEditorModal() {
        const modalHTML = `
            <span class="cnet-modal-close">&times;</span>
            <div class="cnet-modal-content">
                <iframe src="${EDITOR_URL}?theme=dark"></iframe>
            </div>
        `;

        const modal = document.createElement('div');
        modal.classList.add('cnet-modal');
        modal.innerHTML = modalHTML;
        modal.hidden = true;
        document.body.appendChild(modal);

        document.querySelector('.cnet-modal-close').addEventListener('click', () => {
            modal.hidden = true;
        });

        return modal;
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
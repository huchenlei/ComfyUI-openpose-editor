import { app } from "/scripts/app.js";

app.registerExtension({
    name: "huchenlei.OpenPoseEditor",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "huchenlei.OpenPoseEditor") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }

            if (!this.properties) {
                this.properties = {};
                this.properties.savedPose = "";
            }

            this.serialize_widgets = true;

            // Output & widget
            // this.imageWidget = this.widgets.find(w => w.name === "image");
            // this.imageWidget.callback = this.showImage.bind(this);
            // this.imageWidget.disabled = true;

            // Non-serialized widgets
            this.jsonWidget = this.addWidget("text", "", this.properties.savedPose, "savedPose");
            this.jsonWidget.disabled = true;
            this.jsonWidget.serialize = true;

            this.openWidget = this.addWidget("button", "open editor", "image", () => {
                console.log("Hello world");
            });
            this.openWidget.serialize = false;

            // On load if we have a value then render the image
            // The value isnt set immediately so we need to wait a moment
            // No change callbacks seem to be fired on initial setting of the value
            requestAnimationFrame(async () => {
                if (this.imageWidget.value) {
                    await this.setImage(this.imageWidget.value);
                }
            });
        }

        nodeType.prototype.showImage = async function (name) {
            let folder_separator = name.lastIndexOf("/");
            let subfolder = "";
            if (folder_separator > -1) {
                subfolder = name.substring(0, folder_separator);
                name = name.substring(folder_separator + 1);
            }
            const img = await loadImageAsync(`/view?filename=${name}&type=input&subfolder=${subfolder}`);
            this.imgs = [img];
            this.setSizeForImage();
            app.graph.setDirtyCanvas(true);
        }

        nodeType.prototype.setImage = async function (name) {
            this.imageWidget.value = name;
            await this.showImage(name);
        }

        const onPropertyChanged = nodeType.prototype.onPropertyChanged;
        nodeType.prototype.onPropertyChanged = function (property, value) {
            if (property === "savedPose") {
                this.jsonWidget.value = value;
            } else if (onPropertyChanged) {
                onPropertyChanged.apply(this, arguments)
            }
        }
    }
});
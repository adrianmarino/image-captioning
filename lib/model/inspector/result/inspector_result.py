from IPython.display import display, HTML

from lib.utils.plot_utils import show_img


class InspectorResult:
    def __init__(self, image_path, data):
        self.image_path = image_path
        self.data = data

    def show(self, image_width=500):
        show_img(self.image_path, image_width=image_width)
        display(HTML(self.data.to_html(index=False)))

    def wmd_sim(self):
        return self.data['WMDSim Mean (⟰ best)'][0]

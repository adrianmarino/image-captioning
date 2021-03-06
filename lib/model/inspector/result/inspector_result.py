from lib.utils.plot_utils import show_img, display_table


class InspectorResult:
    def __init__(self, image_path, data):
        self.image_path = image_path
        self.data = data

    def show(self, image_width=500):
        show_img(self.image_path, image_width=image_width)
        display_table(self.data)

    def wmd_sim(self):
        return self.data['WMDSim Mean (⟰ best)'][0]

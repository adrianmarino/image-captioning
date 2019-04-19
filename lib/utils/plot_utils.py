import matplotlib.pyplot as plt


def show_sample(image_path, description):
    x = plt.imread(image_path)
    plt.imshow(x)
    plt.show()
    print(description)
class ImagePipeline:
    def __init__(self):
        self.original = None
        self.current = None
        self.history = []
        self.operation_names = []

    def load(self, image):
        self.original = image.copy()
        self.current = image.copy()
        self.history = []
        self.operation_names = []

    def apply(self, image, name):
        if self.current is None:
            raise ValueError("No image loaded")
        self.history.append(self.current.copy())
        self.operation_names.append(name)
        self.current = image.copy()

    def undo(self):
        if not self.history:
            return False
        self.current = self.history.pop()
        if self.operation_names:
            self.operation_names.pop()
        return True

    def reset(self):
        if self.original is None:
            return False
        self.current = self.original.copy()
        self.history = []
        self.operation_names = []
        return True


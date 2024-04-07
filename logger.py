import datetime


class Logger:

    def __init__(self, path: str):
        self.path = path

    def __call__(self, message: str) -> None:
        self.log(message)

    def log(self, message: str) -> None:
        with open(self.path, 'a') as f:
            f.write(f'[{datetime.datetime.now()}] {message}\n')

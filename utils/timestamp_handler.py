import os
from datetime import datetime

class TimestampHandler:
    def __init__(self, file_path: str) -> None:
        assert os.path.exists(file_path)
        self.file_path = file_path
        with open(file_path, 'r') as f:
            self.text = f.read()

    def _covert_to_timestamp(self, timestr: str) -> float:
        dt = datetime.strptime(timestr, '%M:%S.%f')
        return dt.minute * 60 + dt.second + dt.microsecond / 1e6

    def _get_lyric_start_time(self) -> float:
        start_row = self.text.splitlines()[0]
        start_timestamp = self._covert_to_timestamp(start_row[1: 9])

        return start_timestamp

    def get_lyric(self):
        lyric_rows = self.text.splitlines()[1:]
        return ''.join([row.split()[-1] for row in lyric_rows])

    def get_lyric_with_timestamp(self) -> list:
        start_timestamp = self._get_lyric_start_time()

        lyric_rows = self.text.splitlines()[1:]
        lyric_info = []
        for row in lyric_rows:
            row = row.split(' ')
            lyric_info.append({'char': row[-1],
                          'start': self._covert_to_timestamp(row[0][1: -1]) - start_timestamp,
                          'end': self._covert_to_timestamp(row[1][1: -1]) - start_timestamp})


        return lyric_info
            
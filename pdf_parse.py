import os
import PyPDF2
import pprint
import xlwt
import datetime as dt


def set_file_path(file_path: str) -> str:
    """set the file path fro pdfs"""

    if os.getcwd() is not str(file_path):
        os.chdir(file_path)

    return file_path


class ExtractTextFromPDF(object):

    def __init__(self, file_path: str):
        """init method"""
        if os.getcwd() is not str(file_path):
            self._file_path = file_path
            os.chdir(self._file_path)

        self.pdfs_in_memory = list()        # stores names of pdfs that have already been converted to text
        self.users = list()

        self.page_counter = 0
        self._chunk_size = 0

        self._pdfs_dict = dict()
        self._pdf_keys = list()
        self._text_dict = dict()


    def import_pdfs(self):
        """import the pdf names"""
        # --- setup a key,value pair for users and their accompanying pdfs
        self._pdfs_dict = {str(f).split(".")[0]: f for f in os.listdir()}
        self._pdf_keys = list(self._pdfs_dict.keys())

    def convert_pdf_to_text(self, chunk_size: int):
        """convert a specified number of users pdfs into text"""

        self._chunk_size = chunk_size
        iterator = 0
        temp_list = []
        for _pdf in list(self._pdf_keys):
            # --- open pdf
            pdf_object = open(os.path.basename(self._pdfs_dict.get('{}'.format(_pdf))), 'rb')
            # --- call PyPDF2 reader
            pdf_readr = PyPDF2.PdfFileReader(pdf_object)
            # --- number of pages of current pdf
            number_of_pgs = pdf_readr.getNumPages()
            # --- counter for current page of current pdf
            page_cntr = 0
            # --- loop grabs each page of current pdf and converts it to text
            while page_cntr < number_of_pgs:
                page = pdf_readr.getPage(page_cntr)
                # --- text split by line
                pdf_to_text = list(str(page.extractText()).splitlines())
                # --- text added to a temporary list
                temp_list.append(pdf_to_text)
                page_cntr += 1
            pdf_object.close()
            # --- update text dictionary with user information
            # --- and the corresponding text pdf data
            self._text_dict.update({str(_pdf): temp_list})
            # --- used to keep track  of stored user data so as not to read in thir data twice
            self.users.append(str(_pdf))
            iterator += 1
            if iterator == self._chunk_size:
                # --- reset the chunk size
                self._chunk_size = 0
                break















if __name__=="__main__":

    FILE = set_file_path(file_path="C:\\Users\\wmurphy\\Desktop\\relias_pdf")
    pdf_text_readr = ExtractTextFromPDF(file_path=FILE)
    pdf_text_readr.import_pdfs()
    pdf_text_readr.convert_pdf_to_text(chunk_size=4)
    pprint.pprint(pdf_text_readr._text_dict)


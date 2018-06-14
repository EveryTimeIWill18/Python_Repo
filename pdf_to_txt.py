import os
import PyPDF2
import pprint
import xlwt
import datetime as dt
import ExcelMetaData
import re
from itertools import chain
import pandas as pd



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

        self.excel_workbook = None
        self.excel_row = 1
        self.excel_col = 1


    def import_pdfs(self):
        """import the pdf names"""
        # --- setup a key,value pair for users and their accompanying pdfs
        self._pdfs_dict = {str(f).split(".")[0]: f for f in os.listdir()}
        self._pdf_keys = list(self._pdfs_dict.keys())

    def convert_pdf_to_text(self, chunk_size: int):
        """convert a specified number of users pdfs into text"""

        self._chunk_size = chunk_size
        iterator = 0

        for _pdf in list(self._pdf_keys):
            temp_list = []
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
            # --- update text dictionary with user information and the corresponding text pdf data
            self._text_dict.update({str(_pdf): temp_list})
            # --- used to keep track  of stored user data so as not to read in their data twice
            self.users.append(str(_pdf))
            iterator += 1
            if iterator == self._chunk_size:
                # --- reset the chunk size
                self._chunk_size = 0
                break

    def create_excel_workbook(self, sheet_name: str, save: bool, save_path=None, workbook_name=None):
        """create an excel sheet to store the data"""
        self.excel_workbook = xlwt.Workbook(encoding="utf-8")
        sheet = self.excel_workbook.add_sheet(str(sheet_name))

        for i, _ in enumerate(list(excel_columns)):
            sheet.write(0, i+1, excel_columns[i])

        for j, _ in enumerate(list(self.users)):
            if j == 0:
                sheet.write(j, 0, "H2H Emp Name")
            else:
                sheet.write(j, 0, self.users[j])
        if save is True:
            self.excel_workbook.save(str(save_path+"\\"+workbook_name+".xls"))


    def get_excel_sheet(self):
        """use pandas to read in excel spreadsheet for updating"""


    def create_excel_output(self,
                            sheet_name: str,
                            save: bool,
                            save_path=None,
                            workbook_name=None,
                            date_sensitive=False,
                            curriculum=None):

        self.excel_workbook = xlwt.Workbook(encoding="utf-8")
        sheet = self.excel_workbook.add_sheet(str(sheet_name))

        users = list(self.users)
        sheet.write(0, 0, "H2H Emp Name")
        user_cntr = 0
        excel_columns = curriculum

        if date_sensitive:
            valid_start = dt.date(2017, 2, 1)
            valid_end = dt.date(2018, 2, 28)
            for i, _ in enumerate(excel_columns):
                # --- create column in worksheet
                sheet.write(0, i+1, excel_columns[i])
                # --- select current column
                current_col = str(excel_columns[i])
                # --- write current h2h user to
                #sheet.write(i+1, 0, users[user_cntr])
                for j, _ in enumerate(users):
                    current_user = users[j]
                    if i == 0:
                        sheet.write(j+1, 0, current_user)
                    current_text_obj = list(chain(*self._text_dict.get(str(current_user))))
                    # Strip empty lines
                    current_text_obj = [x for x in current_text_obj if x]
                    # Concatenate list into single string
                    current_text_obj = ''.join(current_text_obj).lower()
                    # Verify that text of current column name appears.
                    pattern = (r"(" + current_col + r")"
                               r"(Event|Course){,1}"
                               r"(100|[2-9][0-9]){,1}"
                               r"(\d{1,2}/\d{1,2}/\d{4}){1,2}")
                    matches = re.finditer(pattern,
                                          current_text_obj,
                                          flags=re.IGNORECASE)
                    match = None
                    for match in matches:
                        pass
                    if match:
                        date_completed = match.group(4)
                        month, day, year = map(int,
                                               date_completed.split("/"))
                        date_completed = dt.date(year, month, day)
                        if (date_completed >= valid_start
                                and date_completed <= valid_end):
                            sheet.write(j+1, i+1, 1)

        else:
            for i, _ in enumerate(excel_columns):
                # --- create column in worksheet
                sheet.write(0, i+1, excel_columns[i])
                # --- select current column
                current_col = str(excel_columns[i])
                # --- write current h2h user to
                #sheet.write(i+1, 0, users[user_cntr])
                for j, _ in enumerate(users):
                    current_user = users[j]
                    if i == 0:
                        sheet.write(j+1, 0, current_user)
                    current_text_obj = list(chain(*self._text_dict.get(str(current_user))))
                    # Strip empty lines
                    current_text_obj = [x for x in current_text_obj if x]
                    # Concatenate list into single string
                    current_text_obj = ''.join(current_text_obj).lower()
                    # Verify that text of current column name appears.
                    if current_col.lower() in current_text_obj:
                        print("text object: {}\ni = {}| j = {} ".format(current_col, i, j))
                        sheet.write(j+1, i+1, 1)

        if save is True:
            self.excel_workbook.save(str(save_path + workbook_name + '.xls'))



if __name__=="__main__":

    FILE = set_file_path(file_path="C:\\Users\\jhiggins\\Desktop\\Relias Report"
                                   "\\Facebook Counselors\\")
    pdf_text_readr = ExtractTextFromPDF(file_path=FILE)
    pdf_text_readr.import_pdfs()
    pdf_text_readr.convert_pdf_to_text(chunk_size=200)
    pdf_text_readr.create_excel_output(sheet_name="Relias sheet1", save=True,
                                       save_path="C:\\Users\\jhiggins\\Desktop"
                                                 "\\Relias Report\\Reports\\",
                                       workbook_name="RELIAS FACEBOOK COUNSELORS",
                                       date_sensitive=True,
                                       curriculum=ExcelMetaData.h2h_recurring)
    """pdf_text_readr.create_excel_workbook(sheet_name="Sheet 1", save=True,
                                         save_path="C:\\Users\\wmurphy\\Desktop",
                                         workbook_name="RELIAS")"""

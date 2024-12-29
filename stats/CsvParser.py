#!/usr/bin/env python
import csv

class OpenFile:
    '''
        The input CSV file must be comma delimited and contains only numeric 
        data or empty cells. Each empty cell considers as None.
    '''

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

        with open(self.file_path, 'r') as file:
            self.Rows = list(csv.reader(file))

        self.Cols = self.transpose_matrix(self.Rows)
        
    def GetTableByRows(self) -> list:
        return self.Rows
    
    def GetTableByCols(self) -> list:
        return self.Cols
    
    def transpose_matrix(self, matrix):
        # Check the maximum row length to handle varying row lengths
        max_len = max(len(row) for row in matrix)
        # Pad rows with None to make them of equal length
        padded_matrix = [row + [None] * (max_len - len(row)) for row in matrix]
        # Transpose the padded matrix
        transposed = [[padded_matrix[j][i] for j in range(len(padded_matrix))] for i in range(max_len)]
        # Remove None values if padding was used
        return [[element for element in row if element is not None] for row in transposed]

    def GetCol(self, col_id) -> list:
        assert col_id > 0, 'Column numger must be starting from 1'
        output = []
        for row in self.Rows:
            try:
                value = float(row[col_id-1])
                output.append(value)
            except (ValueError, IndexError):
                # Replace non-number cells and empty cells with None
                output.append(None)

        return output


    def GetCols(self, *args) -> list:
        '''
            The input is a list of integers. 
            Output is list of lists of columns.
            Each int represents how many columns to return.
            Eg: input: (3, 2)
                output: list of two lists of columns, first one with [0:2] cols 
                        and second one with [2:3] cols from the original csv.
        '''
        output = []

        for i, arg in enumerate(args):
            output.append([])
            start = sum(args) - sum(args[i:])
            stop = start + arg

            for col_id in range(start, stop):
                output[i].append(self.GetCol(col_id+1))

        return output


    def GetRows(self, *args) -> list:
        '''
            The input is a list of integers. 
            Output is list of matrices.
            Each int represents each output matrix and defines
            how many columns to include in the matrix.
            Eg: input: (3, 2)
                output: list of two matrices, first one with 1-3 cols 
                        and second one with 4-5 cols from the original csv.
        '''
        output = []

        for i, arg in enumerate(args):
            output.append([])
            start = sum(args) - sum(args[i:])
            stop = start + arg

            for row in self.Rows:
                output[i].append(
                    [float(cell) if cell else None for cell in row[start:stop]]
                )

        return output

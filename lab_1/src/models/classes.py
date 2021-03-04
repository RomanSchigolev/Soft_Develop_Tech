from math import (sin, cos, tan)
import re

import numpy as np


class Scalar:
    def __init__(self, value):
        if Scalar.__entered_data_validator(value):
            self.__data = float(value)
        else:
            raise TypeError('Это не число')

    @staticmethod
    def __entered_data_validator(value):
        regexp_query = re.compile(r'[+-]?(([1-9][0-9]*)|(0))([.,][0-9]+)?')
        return regexp_query.fullmatch(value)

    @property
    def get_value(self):
        return self.__data

    @property
    def get_inversion(self):
        return -self.__data

    @property
    def get_sinus(self):
        return sin(self.__data)

    @property
    def get_cosine(self):
        return cos(self.__data)

    @property
    def get_tangent(self):
        return tan(self.__data)

    @property
    def get_cotangent(self):
        return self.get_cosine / self.get_sinus

    def __add__(self, another):
        if isinstance(another, Scalar):
            return self.__data + another.get_value
        return self.__data + another

    def __mul__(self, another):
        if isinstance(another, Scalar):
            return self.__data * another.get_value
        return self.__data * another

    def get_exponentiation(self, number):
        return pow(self.get_value, float(number))

    def get_root_of_n_degree(self, number):
        return pow(self.get_value, 1 / float(number))


class Vector:
    def __init__(self, data):
        if Vector.__entered_data_validator(data) and len(data) != 0:
            modified_data = [float(i) for i in data.split(', ')]
            self.__data = np.array(modified_data)
        elif len(data) == 0:
            self.__data = np.zeros(2)
        else:
            raise TypeError('Введенные данные не яляются числами')

    @staticmethod
    def __entered_data_validator(entered_data):
        modified_data = [i for i in entered_data.split(', ')]
        regexp_query = re.compile(r'[+-]?(([1-9][0-9]*)|(0))([.,][0-9]+)?')
        counter_correct_items = 0
        for item in modified_data:
            if regexp_query.fullmatch(item):
                counter_correct_items += 1
        return len(modified_data) == counter_correct_items

    @property
    def get_numpy_vector(self):
        return self.__data

    @property
    def get_vector_len(self):
        return len(self.__data)

    @property
    def get_module_vector(self):
        return np.linalg.norm(self.__data)

    def __add__(self, another):
        if not isinstance(another, Vector):
            raise TypeError('Это не вектор')
        if self.get_vector_len != another.get_vector_len:
            raise ValueError('Разные длины векторов')
        return self.__data + another.get_numpy_vector

    def __mul__(self, another):
        if isinstance(another, (int, float)):
            return self.__data * another
        if isinstance(another, Scalar):
            return self.__data * another.get_value
        if isinstance(another, Vector):
            if self.get_vector_len != another.get_vector_len:
                raise ValueError('Разные длины векторов')
            return self.__data * another.get_numpy_vector

    def scalar_mul_of_vectors(self, another):
        if not isinstance(another, Vector):
            raise TypeError('Это не вектор')
        if self.get_vector_len != another.get_vector_len:
            raise ValueError('Разные длины векторов')
        return np.dot(self.__data, another.get_numpy_vector)

    def checking_for_orthogonality(self, second_vector):
        if not isinstance(second_vector, Vector):
            raise TypeError('Это не вектор')
        return self.scalar_mul_of_vectors(second_vector) == 0

    def vector_by_matrix(self, matrix):
        if not isinstance(matrix, Matrix):
            raise TypeError('Это не матрица')
        if self.get_vector_len != matrix.get_rows:
            raise ValueError(
                'Число столбцов в матрице должно совпадать с числом строк в векторе-столбце')
        return np.dot(self.get_numpy_vector, matrix.get_numpy_matrix)

    def vector_product(self, another):
        if not isinstance(another, Vector):
            raise TypeError('Это не вектор')
        if self.get_vector_len < 3 or another.get_vector_len < 3:
            raise ValueError('Вектор не трехмерный')
        return np.cross(self.__data, another.get_numpy_vector)

    def __checking_collinearity_vectors(self, another):
        if not isinstance(another, Vector):
            raise TypeError('Это не вектор')
        if not np.any(self.get_numpy_vector) or not np.any(another.get_numpy_vector):
            raise ValueError(
                'Один из векторов равен нулю, поэтому вопрос о коллинеарности векторов некорректен!')
        if self.get_vector_len != another.get_vector_len:
            raise ValueError('Разные длины векторов')

        first_vector = self.__data
        second_vector = another.get_numpy_vector
        set_coordinate_relations = []

        if len(first_vector[first_vector > 0]) == self.get_vector_len and \
                len(second_vector[second_vector > 0]) == len(second_vector) or \
                np.count_nonzero(first_vector) != self.get_vector_len and \
                len(second_vector[second_vector > 0]) == len(second_vector):

            for value_first_vector, value_second_vector in zip(first_vector, second_vector):
                set_coordinate_relations.append(
                    value_first_vector / value_second_vector)

            return len(set(set_coordinate_relations)) == 1

        if len(second_vector[second_vector > 0]) != len(second_vector):
            index_nonzero_element = np.where(first_vector != 0)[0][0]
            scalar = second_vector[index_nonzero_element] / first_vector[index_nonzero_element]
            intermediate_vector = first_vector * scalar

            return np.array_equal(intermediate_vector, second_vector)

    def checking_codirectionality_vectors(self, another_vector):
        if self.__checking_collinearity_vectors(another_vector):
            return self.scalar_mul_of_vectors(another_vector) > 0
        return False


class Matrix:
    def __init__(self, data):
        if Matrix.__entered_data_validator(data):
            self.__data = np.matrix(data)
        else:
            raise TypeError('Неправильная матрица')

    @staticmethod
    def __entered_data_validator(matrix):
        if len(matrix) != 0 and len(matrix[0]) != 0:
            for row in matrix:
                if len(row) != len(matrix[0]):
                    return False
                for element in row:
                    if not isinstance(element, (int, float)):
                        return False
            return True
        return False

    @property
    def get_rows(self):
        return self.__data.shape[0]

    @property
    def get_cols(self):
        return self.__data.shape[1]

    @property
    def get_numpy_matrix(self):
        return self.__data.A

    @property
    def get_dimension_of_matrix(self):
        return self.get_numpy_matrix.shape

    @property
    def get_trace(self):
        return np.trace(self.get_numpy_matrix)

    @property
    def get_determinant(self):
        return np.linalg.det(self.get_numpy_matrix)

    @property
    def get_inverse_matrix(self):
        return np.linalg.inv(self.get_numpy_matrix)

    @property
    def get_transpose_matrix(self):
        return self.get_numpy_matrix.T

    def __mul__(self, another):
        if isinstance(another, Scalar):
            return self.get_numpy_matrix.dot(another.get_value)
        if isinstance(another, Matrix):
            if self.get_dimension_of_matrix != another.get_dimension_of_matrix:
                raise ValueError('Размеры матриц должны совпадать')
            return np.multiply(self.get_numpy_matrix, another.get_numpy_matrix)
        else:
            return self.get_numpy_matrix * another

    def __add__(self, another):
        if not isinstance(another, Matrix):
            raise TypeError('Это не матрица')
        if self.get_dimension_of_matrix != another.get_dimension_of_matrix:
            raise ValueError('Размеры матриц должны совпадать')
        return self.get_numpy_matrix + another.get_numpy_matrix

    def matrix_product(self, another):
        if not isinstance(another, Matrix):
            raise TypeError('Это не матрица')
        if self.get_cols != another.get_rows:
            raise ValueError('Количество столбцов первой матрицы не равно количеству строк второй матрицы')
        return np.dot(self.get_numpy_matrix, another.get_numpy_matrix)

    def vector_by_matrix(self, another):
        if not isinstance(another, Vector):
            raise TypeError('Это не вектор')
        if another.get_vector_len != self.get_rows:
            raise ValueError('Разная длина')
        return np.dot(another.get_numpy_vector, self.get_numpy_matrix)

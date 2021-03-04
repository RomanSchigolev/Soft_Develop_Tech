#!/usr/bin/env python

from models.classes import Vector, Scalar, Matrix
from utilities.parsing import matrix_parsing

import click


@click.group()
def cli():
    '''Лаба 1'''
    pass


@cli.group()
def scalar():
    '''Операции по отношению к скалярам'''
    pass


@scalar.command()
@click.option('--scalar1', prompt='Введите первый скаляр')
@click.option('--scalar2', prompt='Введите второй скаляр')
def addition(scalar1, scalar2):
    '''Сумма двух скаляров'''
    try:
        first_scalar, second_scalar = Scalar(scalar1), Scalar(scalar2)
        result = first_scalar + second_scalar
        click.echo(result)
    except TypeError:
        click.echo('Ошибка')


@scalar.command()
@click.option('--scal', prompt='Введите скаляр')
def inversion(scal):
    '''Инверсия скаляра'''
    try:
        first_scalar = Scalar(scal)
        result = first_scalar.get_inversion
        click.echo(result)
    except TypeError:
        click.echo('Ошибка')


@scalar.command()
@click.option('--scalar1', prompt='Введите первый скаляр')
@click.option('--scalar2', prompt='Введите второй скаляр')
def multiplication(scalar1, scalar2):
    '''Произведение двух скаляров'''
    try:
        first_scalar, second_scalar = Scalar(scalar1), Scalar(scalar2)
        result = first_scalar * second_scalar
        click.echo(result)
    except TypeError:
        click.echo('Ошибка')


@scalar.command()
@click.option('--scal', prompt='Введите скаляр')
@click.option('--number', prompt='Введите число')
def exponentiation(scal, number):
    '''Возведение в степень'''
    try:
        first_scalar, exponentiation_to_number = Scalar(scal), number
        result = first_scalar.get_exponentiation(exponentiation_to_number)
        click.echo(result)
    except (TypeError, ValueError):
        click.echo('Ошибка')


@scalar.command()
@click.option('--scal', prompt='Введите скаляр')
@click.option('--number', prompt='Введите число')
def root(scal, number):
    '''Вычисление корня'''
    try:
        first_scalar, degree_root = Scalar(scal), number
        result = first_scalar.get_root_of_n_degree(degree_root)
        click.echo(result)
    except (TypeError, ValueError):
        click.echo('Ошибка')


@scalar.command()
@click.option('--scal', prompt='Введите скаляр')
def sinus(scal):
    '''Синус'''
    try:
        first_scalar = Scalar(scal)
        result = first_scalar.get_sinus
        click.echo(result)
    except TypeError:
        click.echo('Ошибка')


@scalar.command()
@click.option('--scal', prompt='Введите скаляр')
def cosine(scal):
    '''Косинус'''
    try:
        first_scalar = Scalar(scal)
        result = first_scalar.get_cosine
        click.echo(result)
    except TypeError:
        click.echo('Ошибка')


@scalar.command()
@click.option('--scal', prompt='Введите скаляр')
def tangent(scal):
    '''Тангенс'''
    try:
        first_scalar = Scalar(scal)
        result = first_scalar.get_tangent
        click.echo(result)
    except TypeError:
        click.echo('Ошибка')


@scalar.command()
@click.option('--scal', prompt='Введите скаляр')
def cotangent(scal):
    '''Котангенс'''
    try:
        first_scalar = Scalar(scal)
        result = first_scalar.get_cotangent
        click.echo(result)
    except (TypeError, ZeroDivisionError):
        click.echo('Ошибка')


@cli.group()
def vector():
    '''Операции по отношению к векторам'''
    pass


@vector.command()
@click.option('--vec', prompt='Введите вектор. Пример: 1, 2, 3')
@click.option('--scal', prompt='Введите скаляр')
def vec_by_scalar(vec, scal):
    '''Умножение вектора на скаляр'''
    try:
        first_vector, second_scalar = Vector(vec), Scalar(scal)
        result = first_vector * second_scalar
        click.echo(result)
    except TypeError:
        click.echo('Ошибка')


@vector.command()
@click.option('--vector1', prompt='Введите первый вектор. Пример: 1, 2, 3')
@click.option('--vector2', prompt='Введите второй вектор. Пример: 1, 2, 3')
def el_by_el_add(vector1, vector2):
    '''Поэлементное сложение'''
    try:
        first_vector, second_vector = Vector(vector1), Vector(vector2)
        result = first_vector + second_vector
        click.echo(result)
    except TypeError:
        click.echo('Ошибка')


@vector.command()
@click.option('--vector1', prompt='Введите первый вектор. Пример: 1, 2, 3')
@click.option('--vector2', prompt='Введите второй вектор. Пример: 1, 2, 3')
def el_by_el_mul(vector1, vector2):
    '''Поэлементное умножение'''
    try:
        first_vector, second_vector = Vector(vector1), Vector(vector2)
        result = first_vector * second_vector
        click.echo(result)
    except TypeError:
        click.echo('Ошибка')
    except ValueError:
        click.echo('Разная длина векторов')


@vector.command()
@click.option('--vec', prompt='Введите вектор. Пример: 1, 2, 3')
@click.option('--rows', prompt='Введите количество строк для матрицы', type=int)
@click.option('--cols', prompt='Введите количество столбцов для матрицы', type=int)
def vec_by_matrix(vec, rows, cols):
    '''Умножение вектора на матрицу'''
    number_of_rows = rows
    number_of_cols = cols
    try:
        vec = Vector(vec)
        mtrx = Matrix(matrix_parsing(number_of_rows, number_of_cols))
        result = vec.vector_by_matrix(mtrx)
        click.echo(result)
    except ValueError:
        click.echo('Ошибка')
    except TypeError:
        click.echo('Возникла ошибка')


@vector.command()
@click.option('--vector1', prompt='Введите первый вектор. Пример: 1, 2, 3')
@click.option('--vector2', prompt='Введите второй вектор. Пример: 1, 2, 3')
def scalar_product(vector1, vector2):
    '''Скалярное произведение'''
    try:
        first_vector, second_vector = Vector(vector1), Vector(vector2)
        result = first_vector.scalar_mul_of_vectors(second_vector)
        click.echo(result)
    except TypeError:
        click.echo('Ошибка')
    except ValueError:
        click.echo('Разная длина векторов')


@vector.command()
@click.option('--vector1', prompt='Введите первый вектор. Пример: 1, 2, 3')
@click.option('--vector2', prompt='Введите второй вектор. Пример: 1, 2, 3')
def vector_product(vector1, vector2):
    '''Векторное произведение'''
    try:
        first_vector, second_vector = Vector(vector1), Vector(vector2)
        result = first_vector.vector_product(second_vector)
        click.echo(result)
    except TypeError:
        click.echo('Ошибка')
    except ValueError:
        click.echo('Вектор не трехмерный')


@vector.command()
@click.option('--vector1', prompt='Введите вектор. Пример: 1, 2, 3')
def module_vector(vector1):
    '''Вычисление длины (модуля) вектора'''
    try:
        first_vector = Vector(vector1)
        result = first_vector.get_module_vector
        click.echo(result)
    except TypeError:
        click.echo('Ошибка')


@vector.command()
@click.option('--vector1', prompt='Введите первый вектор. Пример: 1, 2, 3')
@click.option('--vector2', prompt='Введите второй вектор. Пример: 1, 2, 3')
def codirect_vectors(vector1, vector2):
    '''Проверка сонаправленности векторов'''
    try:
        first_vector, second_vector = Vector(vector1), Vector(vector2)
        result = first_vector.checking_codirectionality_vectors(second_vector)
        click.echo('Сонаправлены' if result else 'Несонаправлены')
    except ValueError:
        click.echo('Ошибка. Векторы не должны быть нулевыми и должны иметь одинаковую длину')
    except TypeError:
        click.echo('Ошибка')


@vector.command()
@click.option('--vector1', prompt='Введите первый вектор. Пример: 1, 2, 3')
@click.option('--vector2', prompt='Введите второй вектор. Пример: 1, 2, 3')
def orthog_vectors(vector1, vector2):
    '''Проверка векторов на ортогональность'''
    try:
        first_vector, second_vector = Vector(vector1), Vector(vector2)
        result = first_vector.checking_for_orthogonality(second_vector)
        click.echo('Ортогональны' if result else 'Неортогональны')
    except ValueError:
        click.echo('Разная длина векторов')
    except TypeError:
        click.echo('Ошибка')


@cli.group()
def matrix():
    '''Операции по отношению к матрицам'''


@matrix.command()
@click.option('--scal', prompt='Введите скаляр')
@click.option('--rows', prompt='Введите количество строк для матрицы', type=int)
@click.option('--cols', prompt='Введите количество столбцов для матрицы', type=int)
def matrix_by_scalar(scal, rows, cols):
    '''Умножение матрицы на скаляр'''
    number_of_rows = rows
    number_of_cols = cols
    try:
        scal = Scalar(scal)
        mtrx = Matrix(matrix_parsing(number_of_rows, number_of_cols))
        result = mtrx * scal
        click.echo(result)
    except ValueError:
        click.echo('Ошибка')
    except TypeError:
        click.echo('Возникла ошибка. Скаляр - число')


@matrix.command()
@click.option('--rows1', prompt='Введите количество строк для первой матрицы', type=int)
@click.option('--cols1', prompt='Введите количество столбцов для первой матрицы', type=int)
@click.option('--rows2', prompt='Введите количество строк для второй матрицы', type=int)
@click.option('--cols2', prompt='Введите количество столбцов для второй матрицы', type=int)
def el_by_el_add(rows1, cols1, rows2, cols2):
    '''Поэлементное сложение'''
    number_of_rows_for_first_matrix, number_of_cols_for_first_matrix = rows1, cols1
    number_of_rows_for_second_matrix, number_of_cols_for_second_matrix = rows2, cols2
    try:
        first_matrix = Matrix(matrix_parsing(number_of_rows_for_first_matrix, number_of_cols_for_first_matrix))
        second_matrix = Matrix(matrix_parsing(number_of_rows_for_second_matrix, number_of_cols_for_second_matrix))
        result = first_matrix + second_matrix
        click.echo(result)
    except ValueError:
        click.echo('Размеры матриц должны совпадать')
    except TypeError:
        click.echo('Это не матрица')


@matrix.command()
@click.option('--rows1', prompt='Введите количество строк для первой матрицы', type=int)
@click.option('--cols1', prompt='Введите количество столбцов для первой матрицы', type=int)
@click.option('--rows2', prompt='Введите количество строк для второй матрицы', type=int)
@click.option('--cols2', prompt='Введите количество столбцов для второй матрицы', type=int)
def el_by_el_mul(rows1, cols1, rows2, cols2):
    '''Поэлементное произведение'''
    number_of_rows_for_first_matrix, number_of_cols_for_first_matrix = rows1, cols1
    number_of_rows_for_second_matrix, number_of_cols_for_second_matrix = rows2, cols2
    try:
        first_matrix = Matrix(matrix_parsing(number_of_rows_for_first_matrix, number_of_cols_for_first_matrix))
        second_matrix = Matrix(matrix_parsing(number_of_rows_for_second_matrix, number_of_cols_for_second_matrix))
        result = first_matrix * second_matrix
        click.echo(result)
    except ValueError:
        click.echo('Размеры матриц должны совпадать')
    except TypeError:
        click.echo('Это не матрица')


@matrix.command()
@click.option('--vec', prompt='Введите вектор. Пример: 1, 2, 3')
@click.option('--rows', prompt='Введите количество строк для матрицы', type=int)
@click.option('--cols', prompt='Введите количество столбцов для матрицы', type=int)
def vec_by_matrix(vec, rows, cols):
    '''Умножение вектора на матрицу'''
    number_of_rows = rows
    number_of_cols = cols
    try:
        vec = Vector(vec)
        mtrx = Matrix(matrix_parsing(number_of_rows, number_of_cols))
        result = vec.vector_by_matrix(mtrx)
        click.echo(result)
    except ValueError:
        click.echo('Ошибка')
    except TypeError:
        click.echo('Возникла ошибка')


@matrix.command()
@click.option('--rows1', prompt='Введите количество строк для первой матрицы', type=int)
@click.option('--cols1', prompt='Введите количество столбцов для первой матрицы', type=int)
@click.option('--rows2', prompt='Введите количество строк для второй матрицы', type=int)
@click.option('--cols2', prompt='Введите количество столбцов для второй матрицы', type=int)
def matrix_product(rows1, cols1, rows2, cols2):
    '''Матричное произведение'''
    number_of_rows_for_first_matrix, number_of_cols_for_first_matrix = rows1, cols1
    number_of_rows_for_second_matrix, number_of_cols_for_second_matrix = rows2, cols2
    try:
        first_matrix = Matrix(matrix_parsing(number_of_rows_for_first_matrix, number_of_cols_for_first_matrix))
        second_matrix = Matrix(matrix_parsing(number_of_rows_for_second_matrix, number_of_cols_for_second_matrix))
        result = first_matrix.matrix_product(second_matrix)
        click.echo(result)
    except ValueError:
        click.echo('Количество столбцов первой матрицы не равно количеству строк второй матрицы')
    except TypeError:
        click.echo('Это не матрица')


@matrix.command()
@click.option('--rows', prompt='Введите количество строк для матрицы', type=int)
@click.option('--cols', prompt='Введите количество столбцов для матрицы', type=int)
def trace(rows, cols):
    '''Вычисление следа'''
    number_of_rows, number_of_cols = rows, cols
    try:
        mtrx = Matrix(matrix_parsing(number_of_rows, number_of_cols))
        result = mtrx.get_trace
        click.echo(result)
    except ValueError:
        click.echo('Количество столбцов первой матрицы не равно количеству строк второй матрицы')
    except TypeError:
        click.echo('Возникла ошибка')


@matrix.command()
@click.option('--rows', prompt='Введите количество строк для матрицы', type=int)
@click.option('--cols', prompt='Введите количество столбцов для матрицы', type=int)
def determinant(rows, cols):
    '''Вычисление определителя'''
    number_of_rows, number_of_cols = rows, cols
    if number_of_rows != number_of_cols:
        click.echo('Количество строк должно совпадать с количеством столбцов')
        return
    try:
        mtrx = Matrix(matrix_parsing(number_of_rows, number_of_cols))
        result = mtrx.get_determinant
        click.echo(result)
    except ValueError:
        click.echo('Ошибка')
    except TypeError:
        click.echo('Возникла ошибка')


@matrix.command()
@click.option('--rows', prompt='Введите количество строк для матрицы', type=int)
@click.option('--cols', prompt='Введите количество столбцов для матрицы', type=int)
def inverse_matrix(rows, cols):
    '''Вычисление обратной матрицы'''
    number_of_rows, number_of_cols = rows, cols
    if number_of_rows != number_of_cols:
        click.echo('Количество строк должно совпадать с количеством столбцов')
        return
    try:
        mtrx = Matrix(matrix_parsing(number_of_rows, number_of_cols))
        result = mtrx.get_inverse_matrix
        click.echo(result)
    except ValueError:
        click.echo('Ошибка')
    except TypeError:
        click.echo('Возникла ошибка')


@matrix.command()
@click.option('--rows', prompt='Введите количество строк для матрицы', type=int)
@click.option('--cols', prompt='Введите количество столбцов для матрицы', type=int)
def transpose_matrix(rows, cols):
    '''Транспонирование'''
    number_of_rows, number_of_cols = rows, cols
    try:
        mtrx = Matrix(matrix_parsing(number_of_rows, number_of_cols))
        result = mtrx.get_transpose_matrix
        click.echo(result)
    except ValueError:
        click.echo('Ошибка')
    except TypeError:
        click.echo('Возникла ошибка')


if __name__ == '__main__':
    cli()

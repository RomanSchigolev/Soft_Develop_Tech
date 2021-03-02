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
@click.option('--vector1', prompt='Введите первый вектор. Пример: 1, 2, 3')
@click.option('--vector2', prompt='Введите второй вектор. Пример: 1, 2, 3')
def vec_by_matrix():
    '''Умножение вектора на матрицу'''
    pass


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
@click.option('--rows', prompt='Введите количество строк', type=int)
@click.option('--cols', prompt='Введите количество столбцов', type=int)
def test(rows, cols):
    number_of_rows = rows
    number_of_cols = cols
    try:
        first_matrix = Matrix(matrix_parsing(number_of_rows, number_of_cols))
    except ValueError:
        click.echo('Ошибка')


if __name__ == '__main__':
    cli()

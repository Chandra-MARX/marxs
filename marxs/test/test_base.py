from ..base import DocMeta

def test_docsting_inheritance():
    class A(object):
        '''class doc here'''
        __metaclass__ = DocMeta

        A = 5 # int: attribute A

        def func(a, b):
            '''Function that does stuff with a and b.'''
            pass

    class B(A):
        def func(a,b,c):
            pass

    assert 'Function that does stuff' in B.func.__doc__

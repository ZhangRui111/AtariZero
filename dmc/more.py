import copy


def assignment_copy_deepcopy():
    """ Take care of the difference of assignment, copy and deepcopy. """
    print("不可变对象:")
    # 不可变对象类型，没有被拷贝的说法，或者说，不可变类型，不管是深拷贝还是浅拷贝，
    # 地址值和拷贝后的值都是一样的。
    a1 = (1, 2, 3)
    print("===== 赋值 =====")
    b1 = a1
    print(a1)
    print(b1)
    print(id(a1))
    print(id(b1))
    print("===== 浅拷贝 =====")
    b2 = copy.copy(a1)
    print(a1)
    print(b2)
    print(id(a1))
    print(id(b2))
    print("===== 深拷贝 =====")
    b3 = copy.deepcopy(a1)
    print(a1)
    print(b3)
    print(id(a1))
    print(id(b3))

    print("\n可变对象:")
    # 赋值： 值相等，地址相等
    # copy浅拷贝：值相等，地址不相等
    # deepcopy深拷贝：值相等，地址不相等
    print("===== 赋值 =====")
    a1 = [1, 2, 3]
    b1 = a1
    print(a1)
    print(b1)
    print(id(a1))
    print(id(b1))
    b1.append(4)
    print("-> b1.append(4)")
    print(a1)
    print(b1)
    print(id(a1))
    print(id(b1))
    print("===== 浅拷贝 =====")
    a2 = [1, 2, 3]
    b2 = copy.copy(a2)
    print(a2)
    print(b2)
    print(id(a2))
    print(id(b2))
    b2.append(4)
    print("-> b2.append(4)")
    print(a2)
    print(b2)
    print(id(a2))
    print(id(b2))
    print("===== 深拷贝 =====")
    a3 = [1, 2, 3]
    b3 = copy.deepcopy(a3)
    print(a3)
    print(b3)
    print(id(a3))
    print(id(b3))
    b3.append(4)
    print("-> b3.append(4)")
    print(a3)
    print(b3)
    print(id(a3))
    print(id(b3))

    # ------------------------------------
    # 1.赋值对象随着原列表一起变化。
    # (赋值并不会产生一个独立的对象单独存在)
    # 2.外层添加元素时，浅拷贝不会随原列表变化而变化；内层添加元素时，浅拷贝才会变化。
    # (浅拷贝之所以称为浅拷贝，是它仅仅只拷贝了一层，拷贝了最外围的对象本身，内部的元素都只是拷贝了一个引用而已)
    # 3.无论原列表如何变化，深拷贝都保持不变。
    # (深拷贝出来的对象是一个全新的对象，不再与原来的对象有任何关联。)
    # ------------------------------------
    print("\n可变对象(外层改变元素):")
    l = [1, 2, 3, [4, 5]]
    l1 = l  # 赋值
    l2 = copy.copy(l)  # 浅拷贝
    l3 = copy.deepcopy(l)  # 深拷贝
    print(l)
    print(l1)
    print(l2)
    print(l3)
    print(id(l))
    print(id(1))
    print(id(l2))
    print(id(l3))
    l.append(6)
    print("-> l.append(6)")
    print(l)
    print(l1)
    print(l2)
    print(l3)
    print(id(l))
    print(id(1))
    print(id(l2))
    print(id(l3))

    print("\n可变对象(内层改变元素):")
    l = [1, 2, 3, [4, 5]]
    l1 = l  # 赋值
    l2 = copy.copy(l)  # 浅拷贝
    l3 = copy.deepcopy(l)  # 深拷贝
    print(l)
    print(l1)
    print(l2)
    print(l3)
    print(id(l))
    print(id(1))
    print(id(l2))
    print(id(l3))
    l[3].append(6)
    print("-> l[3].append(6)")
    print(l)
    print(l1)
    print(l2)
    print(l3)
    print(id(l))
    print(id(1))
    print(id(l2))
    print(id(l3))


# if __name__ == '__main__':
#     assignment_copy_deepcopy()

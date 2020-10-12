# Java面向对象

- java类以及类成员
- 面向对象编程（Object Oriented Programming，OOP）三大特性，封装、继承、多肽
- 修饰符. `this, super, abstract, final static interface package`



面向对象和面向过程的区别与联系？

- 二者都是一种思想，面向对象是相对于面向过程而言的。
- 面向过程，强调的是功能行为，以函数为最小单位，考虑怎么做。
- 面向对象，将功能封装进对象，强调具备了功能的对象，以类/对象为最小单位，考虑谁来做。
- 面向对象更加强调运用人类在日常的思维逻辑中采用的思想方法与原则，如抽象、分类、继承、聚合、多态等。



### 一个类中有哪些成员

一个类中的成员有：属性、方法、构造器、代码块、内部类。



### 对象的创建过程











### 1. java访问权限修饰符

java的访问权限修饰符有四种：public、protected、默认、private。



|           | 当前类 | 同一个包 | 子孙类 | 其他包中的类 |
| :-------: | :----: | :------: | :----: | :----------: |
|  public   |   是   |    是    |   是   |      是      |
| protected |   是   |    是    |   是   |      否      |
|   默认    |   是   |    是    |   否   |      否      |
|  private  |   是   |    否    |   否   |      否      |



实验，

#### **不同包的情况：**

 只要public修饰的变量可以被访问

```java
package com;

public class Person {
    protected String name;
    public void talk(){
        System.out.println("hello world");
    }
	void hello(){
        System.out.println("你好啊");
    }

}
```



```java
package interview;

import com.Person;
public class Demo1 extends Person{
    public static void main(String[] args) {
//        System.out.println(val());
        Person a = new Person();
        a.talk();
        a.name = "小毛";	//报错，属性name有proted修饰，无法被访问。
        a.hello();		//报错，方法hello()使用模式修饰符，无法被访问。
    }
}
```

#### **同一个包**

```java
package com;

public class Person {
    protected String name;
    public void talk(){
        System.out.println("hello world");
    }
	void hello(){
        System.out.println("你好啊");
    }
}
```



```java
package com;

import Person;
public class Demo1 extends Person{
    public static void main(String[] args) {
//        System.out.println(val());
        Person a = new Person();
        a.talk();
        a.name = "小毛";	//可以被访问
        a.hello();		//可以被访问
    }
}
```

#### **子孙类**，有问题

```java
package interview;

public class Animal {
    String name= "123";
    void talk(){
        System.out.println("我的名字是:"+ name);
    }
}
package interview;

public class Dog extends Animal {
    void mmm(){
        name = "123456";	//？默认权限访问控制符修饰的，也可以被继承，不知道为什么
    }
}

package interview;

public class Demo1{
    public static void main(String[] args) {

        Dog dog = new Dog();
        dog.name = "小黄";
        System.out.println(dog.name);
//        dog.talk();
    }
}

```



### 


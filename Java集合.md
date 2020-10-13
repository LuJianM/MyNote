## Java集合

### 1.说说java常见的集合有哪些？

collection接口和map接口是所有集合类的父接口。

collection接口的子接口有list和set；

map接口的实现类有HashMap、TreeMap、HashTable、CurrentHashMap等

list接口的实现类有ArrayList、LinkedList、Vector和Stack。

set接口的实现类有HashSet、TreeSet、LinkedHashset等。

### 2. HashMap与HashTable的区别？

- hashMap是线程不安全的，HashTable由于加了synchronize关键字，所以是线程安全的。
- hashmap允许K/V是都为null，hashTable的K/V都不允许为空。
- HashMap继承自AbstractMap类，HashTable继承自Dictionary类。



### 3. HashMap的put方法的具体流程？




package algorithm;

import java.util.HashMap;
import java.util.Map;

/**
 * @author 吴嘉烺
 * @description 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
 * 实现 LRUCache 类：
 * LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
 * int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
 * void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。
 * 如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
 * 函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。
 * @date 2023/1/29
 */
public class LRUCache {
    class DequeNode {
        int key;
        int value;
        DequeNode prev;
        DequeNode next;
        public DequeNode() {

        }
        public DequeNode(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }
    private Map<Integer, DequeNode> cache = new HashMap<>();
    private int size;
    private int capacity;
    private DequeNode head;
    private DequeNode tail;

    public LRUCache(int capacity) {
        this.size = 0;
        this.capacity = capacity;
        //伪头部和伪尾部
        head = new DequeNode();
        tail = new DequeNode();
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        DequeNode res = cache.get(key);
        if (res == null) {
            return -1;
        }
        moveToHead(res);
        return res.value;
    }

    public void put(int key, int value) {
        DequeNode node = cache.get(key);
        if (node == null) {
            node = new DequeNode(key, value);
            cache.put(key, node);
            addToHead(node);
            size++;
            if (size > capacity) {
                DequeNode tail = removeTail();
                cache.remove(tail.key);
                size--;
            }
        } else {
            node.value = value;
            moveToHead(node);
        }
    }

    private void addToHead(DequeNode node) {
        //在头部增加即是在head.next增加一个node
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
        node.prev = head;
    }

    private void moveToHead(DequeNode node) {
        //移动到头部即是删除该node，然后在头部增加node
        removeNode(node);
        addToHead(node);
    }

    private void removeNode(DequeNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private DequeNode removeTail() {
        //删除尾部即是删除tail的前一个node
        DequeNode node = tail.prev;
        removeNode(node);
        return node;
    }
}

package algorithm;

import java.util.ArrayList;
import java.util.List;

/**
 * @author 吴嘉烺
 * @description
 * @date 2023/7/14
 */
public class LeetCode75 {

    /**
     * 给你两个字符串 word1 和 word2 。请你从 word1 开始，通过交替添加字母来合并字符串。
     * 如果一个字符串比另一个字符串长，就将多出来的字母追加到合并后字符串的末尾。
     * 返回 合并后的字符串 。
     * 输入：word1 = "abc", word2 = "pqr"
     * 输出："apbqcr"
     * 解释：字符串合并情况如下所示：
     * word1：  a   b   c
     * word2：    p   q   r
     * 合并后：  a p b q c r
     * @param word1
     * @param word2
     * @return
     */
    public String mergeAlternately(String word1, String word2) {
        StringBuilder stringBuilder = new StringBuilder();
        int i = 0;
        int j = 0;
        while (i < word1.length() && j < word2.length()) {
            stringBuilder.append(word1.charAt(i++));
            stringBuilder.append(word2.charAt(j++));
        }
        if (i < word1.length()) {
            stringBuilder.append(word1.substring(i));
        }
        if (j < word2.length()) {
            stringBuilder.append(word2.substring(j));
        }
        return stringBuilder.toString();
    }

    /**
     * 对于字符串 s 和 t，只有在 s = t + ... + t（t 自身连接 1 次或多次）时，我们才认定 “t 能除尽 s”。
     * 给定两个字符串 str1 和 str2 。返回 最长字符串 x，要求满足 x 能除尽 str1 且 x 能除尽 str2 。
     * 输入：str1 = "ABCABC", str2 = "ABC"
     * 输出："ABC"
     * @param str1
     * @param str2
     * @return
     */
    public String gcdOfStrings(String str1, String str2) {
        //如果都能除尽，str1 + str2 会等于 str2 + str1（字符串拼接）
        if (!str1.concat(str2).equals(str2.concat(str1))) {
            return "";
        }
        return str1.substring(0, gcdOfStrings_help(str1.length(), str2.length()));
    }

    /**
     * 找出a,b之间的最大公约数
     * @param a
     * @param b
     * @return
     */
    private int gcdOfStrings_help(int a, int b) {
        int tmp = a % b;
        while (tmp != 0) {
            a = b;
            b = tmp;
            tmp = a % b;
        }
        return b;
    }

    /**
     * 给你一个数组 candies 和一个整数 extraCandies ，其中 candies[i] 代表第 i 个孩子拥有的糖果数目。
     * 对每一个孩子，检查是否存在一种方案，将额外的 extraCandies 个糖果分配给孩子们之后，此孩子有 最多 的糖果。
     * 注意，允许有多个孩子同时拥有 最多 的糖果数目。
     * 输入：candies = [2,3,5,1,3], extraCandies = 3
     * 输出：[true,true,true,false,true]
     * 解释：
     * 孩子 1 有 2 个糖果，如果他得到所有额外的糖果（3个），那么他总共有 5 个糖果，他将成为拥有最多糖果的孩子。
     * 孩子 2 有 3 个糖果，如果他得到至少 2 个额外糖果，那么他将成为拥有最多糖果的孩子。
     * 孩子 3 有 5 个糖果，他已经是拥有最多糖果的孩子。
     * 孩子 4 有 1 个糖果，即使他得到所有额外的糖果，他也只有 4 个糖果，无法成为拥有糖果最多的孩子。
     * 孩子 5 有 3 个糖果，如果他得到至少 2 个额外糖果，那么他将成为拥有最多糖果的孩子。
     * @param candies
     * @param extraCandies
     * @return
     */
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        List<Boolean> res = new ArrayList<>();
        int max = 0;
        for (int num : candies) {
            max = Math.max(max, num);
        }
        for (int num : candies) {
            res.add(num + extraCandies >= max);
        }
        return res;
    }

    /**
     * 假设有一个很长的花坛，一部分地块种植了花，另一部分却没有。可是，花不能种植在相邻的地块上，它们会争夺水源，两者都会死去。
     * 给你一个整数数组 flowerbed 表示花坛，由若干 0 和 1 组成，其中 0 表示没种植花，1 表示种植了花。另有一个数 n ，
     * 能否在不打破种植规则的情况下种入 n 朵花？能则返回 true ，不能则返回 false 。
     * 输入：flowerbed = [1,0,0,0,1], n = 1
     * 输出：true
     * @param flowerbed
     * @param n
     * @return
     */
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int max = 0;
        int countZero = 0;
        for (int i = 0; i < flowerbed.length; i++) {
            if (i == 0 && flowerbed[i] == 0) {
                countZero++;
            }
            if (i == flowerbed.length - 1 && flowerbed[i] == 0) {
                countZero++;
            }
            if (flowerbed[i] == 0) {
                countZero++;
            } else {
                if (countZero >= 3) {
                    max += 1 + (countZero - 3) / 2;
                    if (max >= n) {
                        return true;
                    }
                }
                countZero = 0;
            }
        }
        if (countZero >= 3) {
            max += 1 + (countZero - 3) / 2;
        }
        return max >= n;
    }

    public static void main(String[] args) {
        LeetCode75 leetCode75 = new LeetCode75();
        leetCode75.canPlaceFlowers(new int[]{1, 0, 0, 0, 1, 0, 0}, 2);
    }
}

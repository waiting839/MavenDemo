package algorithm;

/**
 * @author 吴嘉烺
 * @description
 * @date 2023/3/1
 */
public class Demo {

    /**
     * 给你一个字符串 s，找到 s 中最长的回文子串。
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        char[] chars = s.toCharArray();
        boolean[][] dp = new boolean[s.length()][s.length()];
        for(int i = 0; i < s.length(); i++){
            dp[i][i] = true;
        }
        int begin = 0;
        int maxL = 0;
        for (int l = 2; l <= s.length(); l++) {
            for (int i = 0; i < s.length(); i++) {
                int j = i + l - 1;
                if (j >= s.length()) {
                    break;
                }
                if (chars[i] != chars[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }
                if (dp[i][j] && j - i + 1 > maxL) {
                    maxL = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxL);
    }
}

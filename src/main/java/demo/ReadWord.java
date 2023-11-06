package demo;

import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.usermodel.XWPFParagraph;
import org.apache.poi.xwpf.usermodel.XWPFPicture;
import org.apache.poi.xwpf.usermodel.XWPFRun;
import org.apache.poi.xwpf.usermodel.XWPFTable;
import org.apache.poi.xwpf.usermodel.XWPFTableCell;
import org.apache.poi.xwpf.usermodel.XWPFTableRow;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.stream.FileImageOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ReadWord {

    private static final Logger log = LoggerFactory.getLogger(ReadWord.class);

    public static final String FILE_PATH = "D:/新建文件夹/村情管家系统-耕地流出分析表 (7).docx";

    public static final String IMAGES_PATH = "D:/新建文件夹/images/";

    public List<Map<String, Object>> wordToMap() throws IOException{
        List<Map<String, Object>> res = new ArrayList<>();

        try (FileInputStream fis = new FileInputStream(FILE_PATH)) {
            XWPFDocument document = new XWPFDocument(fis);

            // 遍历文档中的每个表格
            List<XWPFTable> tables = document.getTables();
            for (XWPFTable table : tables) {
                // 遍历表格的每一行
                List<XWPFTableRow> rows = table.getRows();
                // 计数器，每8行一组数据
                int i = 0;
                Map<String, Object> map = new HashMap<>();
                for (XWPFTableRow row : rows) {
                    // 获取每一行的单元格数据
                    List<XWPFTableCell> cells = row.getTableCells();
                    if (cells == null) {
                        continue;
                    }
                    // cells只有一个的时候为图片
                    if (cells.size() == 2) {
                        map.put(cells.get(0).getText().trim(), cells.get(1).getText().trim());
                    } else if (cells.size() == 1) {
                        byte[] imagesData = getCellImage(cells.get(0));
                        map.put("图片", imagesData);
                    }
                    i++;
                    // 到第八行的时候放进list，为一组数据，重新初始化计数器i和map
                    if (i == 8) {
                        res.add(map);
                        i = 0;
                        map = new HashMap<>();
                    }
                }
            }
        }
        return res;
    }

    public static byte[] getCellImage(XWPFTableCell cell) {
        List<XWPFParagraph> xwpfParagraphs = cell.getParagraphs();
        if (xwpfParagraphs == null) {
            return null;
        }
        for (XWPFParagraph xwpfParagraph : xwpfParagraphs) {
            List<XWPFRun> xwpfRunList = xwpfParagraph.getRuns();
            if (xwpfRunList == null) {
                return null;
            }
            for (XWPFRun xwpfRun : xwpfRunList) {
                List<XWPFPicture> xwpfPictureList = xwpfRun.getEmbeddedPictures();
                if (xwpfPictureList == null) {
                    return null;
                }
                for (XWPFPicture xwpfPicture : xwpfPictureList) {
//                    // 如果要指定路径把图片保存到磁盘，则解除这个注释
//                    byte2image(xwpfPicture.getPictureData().getData(), IMAGES_PATH + xwpfPicture.getPictureData().getFileName());
                    // 按文档上一行应该只有一张图片，直接返回
                    return xwpfPicture.getPictureData().getData();
                }
            }
        }
        return new byte[0];
    }

    public static void byte2image(byte[] data, String path) {
        if (data.length < 3 || path.isEmpty()) {
            return;
        }
        try (FileImageOutputStream imageOutput = new FileImageOutputStream(new File(path))) {
            imageOutput.write(data, 0, data.length);
            log.info("已生成图片到:{}", path);
        } catch (Exception e) {
            log.error("生成图片出现异常", e);
        }
    }

    public static void main(String[] args) throws IOException {
        ReadWord readWord = new ReadWord();
        readWord.wordToMap();
    }
}

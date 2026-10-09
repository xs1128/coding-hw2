// Render a plain A4 report with embedded Times New Roman fonts on macOS.
// Usage: swift render_report.swift document.json report.pdf
import Foundation
import CoreText
import CoreGraphics
import ImageIO

let input = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
let data = try Data(contentsOf: input)
let document = try JSONSerialization.jsonObject(with: data) as! [String: Any]
let blocks = document["blocks"] as! [[String: Any]]
let pageWidth: CGFloat = 595.276
let pageHeight: CGFloat = 841.890
let margin: CGFloat = 56.7
let width = pageWidth - 2 * margin
var media = CGRect(x: 0, y: 0, width: pageWidth, height: pageHeight)
let consumer = CGDataConsumer(url: output as CFURL)!
let pdf = CGContext(consumer: consumer, mediaBox: &media,
                    [kCGPDFContextTitle: "ozdgemm: Adaptive FP64 GEMM Emulation and HPL",
                     kCGPDFContextCreator: "CoreText report renderer"] as CFDictionary)!
var page = 0
var cursor = margin
let mathKey = NSAttributedString.Key("report.math.asset")
var mathDocuments = [String: CGPDFDocument]()

final class MathAsset: NSObject {
    let path: String
    let width: CGFloat
    let ascent: CGFloat
    let descent: CGFloat
    init(_ metadata: [String: Any], size: CGFloat) {
        path = metadata["path"] as! String
        let scale = size / CGFloat(metadata["fontSize"] as! Double)
        width = CGFloat(metadata["width"] as! Double) * scale
        ascent = CGFloat(metadata["ascent"] as! Double) * scale
        descent = CGFloat(metadata["descent"] as! Double) * scale
    }
}

func drawMath(_ asset: MathAsset, x: CGFloat, baseline: CGFloat) {
    if mathDocuments[asset.path] == nil {
        mathDocuments[asset.path] = CGPDFDocument(URL(fileURLWithPath: asset.path) as CFURL)!
    }
    let sheet = mathDocuments[asset.path]!.page(at: 1)!
    let bounds = sheet.getBoxRect(.mediaBox)
    let height = asset.ascent + asset.descent
    pdf.saveGState()
    pdf.translateBy(x: x, y: baseline - asset.descent)
    pdf.scaleBy(x: asset.width / bounds.width, y: height / bounds.height)
    pdf.translateBy(x: -bounds.minX, y: -bounds.minY)
    pdf.drawPDFPage(sheet)
    pdf.restoreGState()
}

func drawFrame(_ frame: CTFrame) {
    pdf.saveGState()
    pdf.textMatrix = .identity
    CTFrameDraw(frame, pdf)
    pdf.restoreGState()
    let lines = CTFrameGetLines(frame) as! [CTLine]
    var origins = [CGPoint](repeating: .zero, count: lines.count)
    CTFrameGetLineOrigins(frame, CFRange(location: 0, length: 0), &origins)
    let bounds = CTFrameGetPath(frame).boundingBox
    for (index, line) in lines.enumerated() {
        for run in CTLineGetGlyphRuns(line) as! [CTRun] {
            let attrs = CTRunGetAttributes(run) as NSDictionary
            guard let asset = attrs[mathKey.rawValue] as? MathAsset else { continue }
            var position = CGPoint.zero
            CTRunGetPositions(run, CFRange(location: 0, length: 1), &position)
            drawMath(asset, x: bounds.minX + origins[index].x + position.x,
                     baseline: bounds.minY + origins[index].y + position.y)
        }
    }
}

func attributes(_ size: CGFloat, bold: Bool = false, italic: Bool = false,
                leading: CGFloat? = nil, center: Bool = false, right: Bool = false,
                justified: Bool = false) -> [NSAttributedString.Key: Any] {
    let name = bold ? (italic ? "TimesNewRomanPS-BoldItalicMT" : "TimesNewRomanPS-BoldMT")
                    : (italic ? "TimesNewRomanPS-ItalicMT" : "TimesNewRomanPSMT")
    let font = CTFontCreateWithName(name as CFString, size, nil)
    precondition((CTFontCopyPostScriptName(font) as String) == name, "Times New Roman missing")
    var alignment: CTTextAlignment = center ? .center : (right ? .right : (justified ? .justified : .left))
    var lineHeight = leading ?? size * 1.28
    var settings = [CTParagraphStyleSetting]()
    return withUnsafePointer(to: &alignment) { a in
        withUnsafePointer(to: &lineHeight) { l in
            settings = [CTParagraphStyleSetting(spec: .alignment, valueSize: MemoryLayout<CTTextAlignment>.size, value: a),
                        CTParagraphStyleSetting(spec: .minimumLineHeight, valueSize: MemoryLayout<CGFloat>.size, value: l)]
            let style = CTParagraphStyleCreate(settings, settings.count)
            return [NSAttributedString.Key(kCTFontAttributeName as String): font,
                    NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 0, alpha: 1),
                    NSAttributedString.Key(kCTParagraphStyleAttributeName as String): style]
        }
    }
}

func attributed(_ block: [String: Any], size: CGFloat, bold: Bool = false,
                italic: Bool = false, center: Bool = false, justified: Bool = false) -> NSAttributedString {
    let result = NSMutableAttributedString(string: "")
    let runs = block["runs"] as? [[String: Any]] ?? [["text": block["text"] as? String ?? ""]]
    for run in runs {
        if let math = run["math"] as? [String: Any] {
            let asset = MathAsset(math, size: size)
            var callbacks = CTRunDelegateCallbacks(version: kCTRunDelegateVersion1,
                dealloc: { ref in Unmanaged<MathAsset>.fromOpaque(ref).release() },
                getAscent: { ref in Unmanaged<MathAsset>.fromOpaque(ref).takeUnretainedValue().ascent },
                getDescent: { ref in Unmanaged<MathAsset>.fromOpaque(ref).takeUnretainedValue().descent },
                getWidth: { ref in Unmanaged<MathAsset>.fromOpaque(ref).takeUnretainedValue().width })
            let delegate = CTRunDelegateCreate(&callbacks, Unmanaged.passRetained(asset).toOpaque())!
            var attrs = attributes(size, center: center, right: block["right"] as? Bool ?? false,
                                   justified: justified)
            attrs[NSAttributedString.Key(kCTRunDelegateAttributeName as String)] = delegate
            attrs[mathKey] = asset
            attrs[NSAttributedString.Key(kCTForegroundColorAttributeName as String)] = CGColor(gray: 0, alpha: 0)
            result.append(NSAttributedString(string: "\u{FFFC}", attributes: attrs))
            continue
        }
        result.append(NSAttributedString(string: run["text"] as? String ?? "",
            attributes: attributes(size, bold: bold || (run["bold"] as? Bool ?? false),
                                   italic: italic || (run["italic"] as? Bool ?? false), center: center,
                                   right: block["right"] as? Bool ?? false, justified: justified)))
    }
    return result
}

func measure(_ text: NSAttributedString, _ available: CGFloat) -> CGFloat {
    let setter = CTFramesetterCreateWithAttributedString(text)
    let s = CTFramesetterSuggestFrameSizeWithConstraints(setter, CFRange(location: 0, length: text.length), nil,
                                                        CGSize(width: available, height: 10000), nil)
    return ceil(s.height) + 4
}

func draw(_ text: NSAttributedString, x: CGFloat, top: CGFloat, available: CGFloat, height: CGFloat) {
    let path = CGPath(rect: CGRect(x: x, y: pageHeight - top - height, width: available, height: height), transform: nil)
    let frame = CTFramesetterCreateFrame(CTFramesetterCreateWithAttributedString(text),
                                        CFRange(location: 0, length: text.length), path, nil)
    let visible = CTFrameGetVisibleStringRange(frame)
    precondition(visible.length == text.length, "Text would be clipped: \(text.string.prefix(70))")
    drawFrame(frame)
}

func endPage() {
    let n = NSAttributedString(string: "\(page)", attributes: attributes(9, center: true))
    draw(n, x: margin, top: pageHeight - 34, available: width, height: 15)
    pdf.endPDFPage()
}

func newPage() {
    if page > 0 { endPage() }
    pdf.beginPDFPage(nil)
    page += 1
    cursor = margin
}

func ensure(_ height: CGFloat) {
    precondition(height <= pageHeight - 2 * margin, "Block exceeds a page")
    if cursor + height > pageHeight - margin { newPage() }
}

// Keep short tables together. Longer tables need only their caption,
// column headers, and first two data rows before they can continue.
func tableStartHeight(_ block: [String: Any]) -> CGFloat {
    let rows = block["rows"] as! [[[String: Any]]]
    let weights = block["weights"] as! [Double]
    let total = weights.reduce(0, +)
    let columns = weights.map { width * CGFloat($0 / total) }
    let size: CGFloat = weights.count >= 7 ? 8.6 : (weights.count >= 5 ? 9.0 : 9.5)
    let initialRows = rows.count <= 4 ? rows : Array(rows.prefix(3))
    let heights = initialRows.enumerated().map { row, cells in
        zip(cells, columns).map { measure(attributed($0.0, size: size, bold: row == 0), $0.1 - 10) }.max()! + 8
    }
    let caption = attributed(["text": block["caption"] as? String ?? ""], size: 9.5)
    return heights.reduce(0, +) + (caption.length > 0 ? measure(caption, width) + 5 : 0) + 4
}

// Lay out complete paragraphs across page boundaries, with at least two
// lines on either side of a break. Use the original attributed paragraph
// in each frame so continuation lines retain their justified alignment.
func flow(_ text: NSAttributedString, size: CGFloat) {
    let setter = CTFramesetterCreateWithAttributedString(text)
    var offset = 0
    func frame(_ height: CGFloat) -> CTFrame {
        let path = CGPath(rect: CGRect(x: margin, y: pageHeight - cursor - height,
                                      width: width, height: height), transform: nil)
        return CTFramesetterCreateFrame(setter, CFRange(location: offset, length: 0), path, nil)
    }
    while offset < text.length {
        var available = pageHeight - margin - cursor
        var current = frame(available)
        let totalLines = (CTFrameGetLines(frame(10000)) as! [CTLine]).count
        var count = (CTFrameGetLines(current) as! [CTLine]).count
        if count < min(2, totalLines) {
            newPage()
            continue
        }
        if totalLines - count == 1 && count > 2 {
            available -= size * 1.28
            current = frame(available)
            count = (CTFrameGetLines(current) as! [CTLine]).count
        }
        let visible = CTFrameGetVisibleStringRange(current)
        precondition(visible.location == offset && visible.length > 0, "Paragraph pagination failed")
        drawFrame(current)
        let fragment = text.attributedSubstring(from: NSRange(location: offset, length: visible.length))
        cursor += min(available, measure(fragment, width))
        offset += visible.length
        if offset < text.length { newPage() }
    }
    cursor += 4
}

newPage()
for (index, block) in blocks.enumerated() {
    let type = block["type"] as! String
    if type == "rule" {
        ensure(24)
        cursor += 8
        pdf.setStrokeColor(CGColor(gray: 0, alpha: 1)); pdf.setLineWidth(0.7)
        pdf.move(to: CGPoint(x: margin, y: pageHeight - cursor))
        pdf.addLine(to: CGPoint(x: margin + width, y: pageHeight - cursor)); pdf.strokePath()
        cursor += 16
    } else if type == "equation" {
        let asset = MathAsset(block["math"] as! [String: Any], size: 11)
        precondition(asset.width <= width, "Equation exceeds the text width")
        let height = asset.ascent + asset.descent
        ensure(height + 14)
        cursor += 5
        drawMath(asset, x: margin + (width - asset.width) / 2,
                 baseline: pageHeight - cursor - asset.ascent)
        cursor += height + 9
    } else if type == "figure" {
        let url = URL(fileURLWithPath: block["path"] as! String)
        let source = CGImageSourceCreateWithURL(url as CFURL, nil)!
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)!
        // Compact, centered charts; keep the source aspect ratio and readable labels.
        let ratio = CGFloat(image.height) / CGFloat(image.width)
        let imageWidth = min(width * 0.82, 190 / ratio)
        let imageHeight = imageWidth * CGFloat(image.height) / CGFloat(image.width)
        let caption = attributed(["runs": block["caption"]!], size: 9.5, justified: true)
        let capHeight = measure(caption, width)
        ensure(imageHeight + capHeight + 14)
        pdf.draw(image, in: CGRect(x: margin + (width - imageWidth) / 2, y: pageHeight - cursor - imageHeight,
                                  width: imageWidth, height: imageHeight))
        cursor += imageHeight + 4
        draw(caption, x: margin, top: cursor, available: width, height: capHeight)
        cursor += capHeight + 10
    } else if type == "table" {
        let rows = block["rows"] as! [[[String: Any]]]
        let weights = block["weights"] as! [Double]
        let total = weights.reduce(0, +)
        let columns = weights.map { width * CGFloat($0 / total) }
        let size: CGFloat = weights.count >= 7 ? 8.6 : (weights.count >= 5 ? 9.0 : 9.5)
        let rowTexts = rows.enumerated().map { row, cells in
            cells.map { attributed($0, size: size, bold: row == 0) }
        }
        let heights = rowTexts.map { cells in
            zip(cells, columns).map { measure($0.0, $0.1 - 10) }.max()! + 8
        }
        let caption = attributed(["text": block["caption"] as? String ?? ""], size: 9.5)
        let captionHeight = caption.length > 0 ? measure(caption, width) + 5 : 0
        let initialHeights = rows.count <= 4 ? heights : Array(heights.prefix(3))
        ensure(captionHeight + initialHeights.reduce(0, +) + 4)
        if captionHeight > 0 {
            draw(caption, x: margin, top: cursor, available: width, height: captionHeight - 5)
            cursor += captionHeight
        }
        func row(_ i: Int) {
            let h = heights[i]
            var x = margin
            if i == 0 {
                pdf.setStrokeColor(CGColor(gray: 0, alpha: 1)); pdf.setLineWidth(0.7)
                pdf.move(to: CGPoint(x: margin, y: pageHeight - cursor))
                pdf.addLine(to: CGPoint(x: margin + width, y: pageHeight - cursor)); pdf.strokePath()
            }
            for (j, text) in rowTexts[i].enumerated() {
                draw(text, x: x + 5, top: cursor + 4, available: columns[j] - 10, height: h - 8)
                x += columns[j]
            }
            cursor += h
            pdf.setStrokeColor(CGColor(gray: i == 0 || i == rows.count - 1 ? 0 : 0.7, alpha: 1))
            pdf.setLineWidth(i == 0 || i == rows.count - 1 ? 0.6 : 0.3)
            pdf.move(to: CGPoint(x: margin, y: pageHeight - cursor))
            pdf.addLine(to: CGPoint(x: margin + width, y: pageHeight - cursor)); pdf.strokePath()
        }
        row(0)
        for i in 1..<rows.count {
            // Avoid leaving a single data row alone on the next page.
            let needed = rows.count - i == 2 ? heights[i] + heights[i + 1] : heights[i]
            if cursor + needed > pageHeight - margin {
                newPage(); row(0)
            }
            row(i)
        }
        cursor += 10
    } else {
        let heading = type == "h1" || type == "h2" || type == "h3"
        let size: CGFloat = type == "h1" ? 18 : (type == "header-meta" ? 12 : (type == "h2" ? 13 : (type == "h3" ? 11.5 :
                            (type == "caption" || type == "meta" || type == "code" ? 9.5 : 11))))
        let text = attributed(block, size: size, bold: heading, center: type == "h1" || type == "header-meta",
                              justified: type == "p" || type == "meta")
        if type == "p" || type == "meta" {
            flow(text, size: size)
            continue
        }
        let h = measure(text, width)
        let before: CGFloat = type == "h2" ? 10 : (type == "h3" ? 5 : 0)
        var following: CGFloat = heading ? 44 : 0
        if heading {
            var extra: CGFloat = 5
            var j = index + 1
            while j < blocks.count {
                let next = blocks[j]
                let kind = next["type"] as! String
                if kind == "h2" || kind == "h3" {
                    extra += measure(attributed(next, size: kind == "h2" ? 13 : 11.5, bold: true), width)
                             + (kind == "h2" ? 15 : 10)
                    j += 1
                    continue
                }
                if kind == "table" {
                    extra += tableStartHeight(next)
                } else if kind == "code" {
                    extra += measure(attributed(next, size: 9.5), width) + 6
                } else if kind == "p" || kind == "meta" {
                    let nextSize: CGFloat = kind == "p" ? 11 : 9.5
                    extra += min(measure(attributed(next, size: nextSize), width), nextSize * 2.56 + 4)
                }
                break
            }
            if h + before + extra <= pageHeight - 2 * margin { following = max(following, extra) }
        }
        ensure(h + before + following)
        cursor += before
        draw(text, x: margin, top: cursor, available: width, height: h)
        cursor += h + (type == "h1" ? 8 : (type == "header-meta" ? 3 : (heading ? 5 : 6)))
    }
}
endPage()
pdf.closePDF()
print("Rendered \(page) pages to \(output.path)")

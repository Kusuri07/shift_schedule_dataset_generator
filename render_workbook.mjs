import fs from "node:fs/promises";
import path from "node:path";

import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const RENDER_SCALE = 2;
const DAY_COLUMN_WIDTH_PX = 48;
const BODY_ROW_HEIGHT_PX = 28;
const THIN_GRID = { preset: "all", style: "thin", color: "#9A9A9A" };
const GROUP_TEMPLATES = new Set(["compact_summary", "grouped_hospital", "parted_pdf", "ood_name_group_swapped"]);
let TEXT_MASK_MODE = false;


function excelColumnName(index1Based) {
  let result = "";
  let value = index1Based;
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}


function pngDimensions(bytes) {
  if (bytes.length < 24 || bytes[0] !== 0x89 || bytes[1] !== 0x50) {
    throw new Error("artifact-tool did not return a PNG image");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return { width: view.getUint32(16), height: view.getUint32(20) };
}


function applyStyle(range, options = {}) {
  if (TEXT_MASK_MODE) {
    options = {
      ...options,
      fill: "#FFFFFF",
      color: "#000000",
      borders: { preset: "all", style: "thin", color: "#FFFFFF" },
    };
  }
  range.format = {
    fill: options.fill ?? "#FFFFFF",
    font: {
      name: "Malgun Gothic",
      size: options.size ?? 10,
      bold: options.bold ?? false,
      color: options.color ?? "#111111",
    },
    horizontalAlignment: options.horizontalAlignment ?? "center",
    verticalAlignment: "center",
    wrapText: options.wrapText ?? false,
    borders: options.borders ?? THIN_GRID,
  };
}


function setColumnWidth(sheet, columnIndex, widthPx) {
  sheet.getRangeByIndexes(0, columnIndex, 1, 1).format.columnWidthPx = widthPx;
}


function setRowHeight(sheet, rowIndex, heightPx) {
  sheet.getRangeByIndexes(rowIndex, 0, 1, 1).format.rowHeightPx = heightPx;
}


function summaryLabels(templateId) {
  if (templateId === "compact_summary") return ["D", "E", "N", "OFF", "연차"];
  if (templateId === "grouped_hospital") return ["D", "E", "N", "OFF"];
  if (templateId === "ood_summary_left") return ["D", "E", "N", "OFF", "연차"];
  return [];
}


function templateTitleFill(templateId) {
  if (templateId === "parted_pdf") return "#D9EAD3";
  if (templateId === "clean_grid") return "#D9EAF7";
  if (templateId === "highlighted_grid") return "#FFF2CC";
  return "#F8B800";
}


function groupFill(group) {
  const palette = ["#F6E5B9", "#EBCFDB", "#CDE2F2", "#E7E7E7"];
  const match = String(group).match(/\d+/);
  const index = Math.max(0, Number(match?.[0] ?? 1) - 1);
  return palette[index % palette.length];
}


function compactCodeFill(code) {
  const normalized = String(code).toUpperCase();
  if (normalized === "D") return "#DCEBD7";
  if (normalized === "E") return "#F6E0D7";
  if (normalized === "N") return "#DDE5F6";
  if (["OFF", "O", "F", "OF"].includes(normalized)) return "#F7DDE3";
  return "#FFFFFF";
}


function highlightedCodeFill(scheduleId, rowIndex, dayIndex, canonicalCode) {
  const common = new Set(["D", "E", "N", "F", "M", "OFF", "O"]);
  if (common.has(String(canonicalCode).toUpperCase())) return rowIndex % 2 === 0 ? "#FFFFFF" : "#F9FAFA";
  const palette = ["#FFB7DB", "#EAF39A", "#A9DFF0"];
  const key = `${scheduleId}:${rowIndex}:${dayIndex}:${canonicalCode}`;
  let hash = 0;
  for (const char of key) hash = (hash * 31 + char.codePointAt(0)) >>> 0;
  return palette[hash % palette.length];
}


function codeFontSize(code) {
  const length = [...String(code)].length;
  if (length <= 2) return 10;
  if (length <= 4) return 8;
  return 7;
}


function scheduleLayout(schedule) {
  const groupColumn = GROUP_TEMPLATES.has(schedule.template_id);
  const noteRow = new Set(["grouped_hospital", "parted_pdf"]).has(schedule.template_id);
  const nameFirst = schedule.template_id === "ood_name_group_swapped";
  const summaryBeforeDays = schedule.template_id === "ood_summary_left";
  const summaries = summaryLabels(schedule.template_id);
  const headerStartRow = noteRow ? 2 : 1;
  const headerRows = schedule.template_id === "ood_multi_header" ? 3 : 2;
  const bodyStartRow = headerStartRow + headerRows;
  const leadingColumns = groupColumn ? 2 : 1;
  const dayStartColumn = leadingColumns + (summaryBeforeDays ? summaries.length : 0);
  const summaryStartColumn = summaryBeforeDays ? leadingColumns : dayStartColumn + schedule.day_count;
  const totalColumns = leadingColumns + schedule.day_count + summaries.length;
  const totalRows = bodyStartRow + schedule.rows.length;

  const nameColumn = nameFirst ? 0 : groupColumn ? 1 : 0;
  const groupColumnIndex = groupColumn ? (nameFirst ? 1 : 0) : null;
  const columnWidths = Array(totalColumns).fill(DAY_COLUMN_WIDTH_PX);
  columnWidths[nameColumn] = 112;
  if (groupColumn) columnWidths[groupColumnIndex] = schedule.template_id === "compact_summary" ? 64 : 92;
  for (let index = 0; index < summaries.length; index += 1) columnWidths[summaryStartColumn + index] = 54;
  if (schedule.template_id === "ood_irregular_columns") {
    for (let day = 0; day < schedule.day_count; day += 1) columnWidths[dayStartColumn + day] = [38, 46, 58, 43][day % 4];
  }

  const rowHeights = [38];
  if (noteRow) rowHeights.push(28);
  for (let index = 0; index < headerRows; index += 1) rowHeights.push(index === 0 && headerRows === 3 ? 24 : 27);
  for (let index = 0; index < schedule.rows.length; index += 1) rowHeights.push(BODY_ROW_HEIGHT_PX);

  return {
    groupColumn,
    nameColumn,
    groupColumnIndex,
    nameFirst,
    summaryBeforeDays,
    noteRow,
    dayStartColumn,
    summaryStartColumn,
    summaries,
    headerStartRow,
    headerRows,
    bodyStartRow,
    totalColumns,
    totalRows,
    columnWidths,
    rowHeights,
    usedRange: `A1:${excelColumnName(totalColumns)}${totalRows}`,
  };
}


function configureScheduleSheet(sheet, schedule, layout) {
  sheet.showGridLines = false;
  layout.columnWidths.forEach((width, index) => setColumnWidth(sheet, index, width));
  layout.rowHeights.forEach((height, index) => setRowHeight(sheet, index, height));

  const pageBadgeColumns = schedule.page_label ? 2 : 0;
  const titleColumns = layout.totalColumns - pageBadgeColumns;
  sheet.mergeCells(`A1:${excelColumnName(titleColumns)}1`);
  sheet.getRange("A1").values = [[schedule.title]];
  applyStyle(sheet.getRangeByIndexes(0, 0, 1, titleColumns), {
    fill: templateTitleFill(schedule.template_id),
    bold: true,
    size: 15,
    borders: { preset: "outside", style: "medium", color: "#222222" },
  });
  if (pageBadgeColumns) {
    const badgeStartColumn = titleColumns;
    sheet.mergeCells(
      `${excelColumnName(badgeStartColumn + 1)}1:${excelColumnName(layout.totalColumns)}1`,
    );
    sheet.getRangeByIndexes(0, badgeStartColumn, 1, 1).values = [[schedule.page_label]];
    applyStyle(sheet.getRangeByIndexes(0, badgeStartColumn, 1, pageBadgeColumns), {
      fill: "#FFF7D1",
      bold: true,
      color: "#3A2D00",
      size: 9,
      borders: { preset: "outside", style: "medium", color: "#725800" },
    });
  }

  if (layout.noteRow) {
    sheet.mergeCells(`A2:${excelColumnName(layout.totalColumns)}2`);
    sheet.getRange("A2").values = [["공지사항: 합성 데이터 / 실제 직원 정보 아님 / 모든 코드는 학습용 무작위 생성"]];
    applyStyle(sheet.getRangeByIndexes(1, 0, 1, layout.totalColumns), {
      fill: "#FFFFFF",
      size: 9,
      horizontalAlignment: "left",
      borders: { preset: "outside", style: "thin", color: "#777777" },
    });
  }

  const headerRow = layout.headerStartRow;
  if (layout.groupColumn) {
    const groupLetter = excelColumnName(layout.groupColumnIndex + 1);
    const nameLetter = excelColumnName(layout.nameColumn + 1);
    sheet.mergeCells(`${groupLetter}${headerRow + 1}:${groupLetter}${headerRow + layout.headerRows}`);
    sheet.getRangeByIndexes(headerRow, layout.groupColumnIndex, 1, 1).values = [[schedule.template_id === "compact_summary" ? "병동" : "구분"]];
    sheet.mergeCells(`${nameLetter}${headerRow + 1}:${nameLetter}${headerRow + layout.headerRows}`);
    sheet.getRangeByIndexes(headerRow, layout.nameColumn, 1, 1).values = [["성명"]];
  } else {
    sheet.mergeCells(`A${headerRow + 1}:A${headerRow + layout.headerRows}`);
    sheet.getRangeByIndexes(headerRow, 0, 1, 1).values = [["성명"]];
  }

  applyStyle(sheet.getRangeByIndexes(headerRow, 0, layout.headerRows, layout.totalColumns), {
    fill: "#F1F4F5",
    bold: true,
    size: 10,
  });

  const dateRow = headerRow + (layout.headerRows === 3 ? 1 : 0);
  const weekdayRow = dateRow + 1;
  if (layout.headerRows === 3) {
    const firstDay = excelColumnName(layout.dayStartColumn + 1);
    const lastDay = excelColumnName(layout.dayStartColumn + schedule.day_count);
    sheet.mergeCells(`${firstDay}${headerRow + 1}:${lastDay}${headerRow + 1}`);
    sheet.getRangeByIndexes(headerRow, layout.dayStartColumn, 1, 1).values = [[`${schedule.year}년 ${schedule.month}월 근무 일정`]];
    applyStyle(sheet.getRangeByIndexes(headerRow, layout.dayStartColumn, 1, schedule.day_count), { fill: "#DDEBF7", bold: true, size: 10 });
  }

  for (let dayIndex = 0; dayIndex < schedule.day_count; dayIndex += 1) {
    const column = layout.dayStartColumn + dayIndex;
    const weekday = schedule.weekdays[dayIndex];
    const weekendFill = weekday === "일" ? "#FCE8E6" : weekday === "토" ? "#E8F0FE" : "#F6F8F8";
    sheet.getRangeByIndexes(dateRow, column, 1, 1).values = [[dayIndex + 1]];
    sheet.getRangeByIndexes(weekdayRow, column, 1, 1).values = [[weekday]];
    applyStyle(sheet.getRangeByIndexes(dateRow, column, 2, 1), {
      fill: weekendFill,
      bold: true,
      color: weekday === "일" ? "#C5221F" : weekday === "토" ? "#185ABC" : "#222222",
      size: 9,
    });
  }

  const summaryStartColumn = layout.summaryStartColumn;
  layout.summaries.forEach((label, index) => {
    const column = summaryStartColumn + index;
    sheet.mergeCells(`${excelColumnName(column + 1)}${headerRow + 1}:${excelColumnName(column + 1)}${headerRow + layout.headerRows}`);
    sheet.getRangeByIndexes(headerRow, column, 1, 1).values = [[label]];
    applyStyle(sheet.getRangeByIndexes(headerRow, column, layout.headerRows, 1), { fill: "#FAFAFA", bold: true, size: 9 });
  });

  for (let rowIndex = 0; rowIndex < schedule.rows.length; rowIndex += 1) {
    const person = schedule.rows[rowIndex];
    const row = layout.bodyStartRow + rowIndex;
    const alternatingFill = rowIndex % 2 === 0 ? "#FFFFFF" : "#F8FAFA";
    applyStyle(sheet.getRangeByIndexes(row, 0, 1, layout.totalColumns), { fill: alternatingFill, size: 9 });

    if (layout.groupColumn) {
      sheet.getRangeByIndexes(row, layout.groupColumnIndex, 1, 1).values = [[person.group]];
      sheet.getRangeByIndexes(row, layout.nameColumn, 1, 1).values = [[person.name]];
      applyStyle(sheet.getRangeByIndexes(row, layout.groupColumnIndex, 1, 1), { fill: groupFill(person.group), bold: true, size: 10 });
      applyStyle(sheet.getRangeByIndexes(row, layout.nameColumn, 1, 1), {
        fill: schedule.template_id === "parted_pdf" ? groupFill(person.group) : alternatingFill,
        size: 10,
      });
    } else {
      sheet.getRangeByIndexes(row, 0, 1, 1).values = [[person.name]];
      applyStyle(sheet.getRangeByIndexes(row, 0, 1, 1), { fill: alternatingFill, size: 10 });
    }

    for (let dayIndex = 0; dayIndex < schedule.day_count; dayIndex += 1) {
      const column = layout.dayStartColumn + dayIndex;
      const displayCode = person.codes_display[dayIndex];
      const canonicalCode = person.codes_canonical[dayIndex];
      let fill = alternatingFill;
      if (schedule.template_id === "compact_summary") fill = compactCodeFill(canonicalCode);
      if (schedule.template_id === "highlighted_grid") {
        fill = highlightedCodeFill(schedule.schedule_id, rowIndex, dayIndex, canonicalCode);
      }
      const cell = sheet.getRangeByIndexes(row, column, 1, 1);
      cell.values = [[displayCode]];
      applyStyle(cell, { fill, size: codeFontSize(displayCode), wrapText: false });
    }

    if (layout.summaries.length > 0) {
      const firstDay = excelColumnName(layout.dayStartColumn + 1);
      const lastDay = excelColumnName(layout.dayStartColumn + schedule.day_count);
      const excelRow = row + 1;
      const dayRange = `${firstDay}${excelRow}:${lastDay}${excelRow}`;
      const formulas = {
        D: `=COUNTIF(${dayRange},"D")`,
        E: `=COUNTIF(${dayRange},"E")`,
        N: `=COUNTIF(${dayRange},"N")`,
        OFF: `=COUNTIF(${dayRange},"OFF")+COUNTIF(${dayRange},"O")+COUNTIF(${dayRange},"F")+COUNTIF(${dayRange},"OF")`,
        연차: `=COUNTIF(${dayRange},"연차")+COUNTIF(${dayRange},"연가")+COUNTIF(${dayRange},"AL")+COUNTIF(${dayRange},"A/L")`,
      };
      layout.summaries.forEach((label, index) => {
        const cell = sheet.getRangeByIndexes(row, summaryStartColumn + index, 1, 1);
        cell.formulas = [[formulas[label]]];
        applyStyle(cell, { fill: "#FFFFFF", size: 9 });
      });
    }
  }

  if (new Set(["grouped_hospital", "parted_pdf", "ood_name_group_swapped"]).has(schedule.template_id)) {
    let start = 0;
    while (start < schedule.rows.length) {
      let end = start;
      while (end + 1 < schedule.rows.length && schedule.rows[end + 1].group === schedule.rows[start].group) end += 1;
      if (end > start) {
        const firstRow = layout.bodyStartRow + start + 1;
        const lastRow = layout.bodyStartRow + end + 1;
        const groupLetter = excelColumnName(layout.groupColumnIndex + 1);
        sheet.mergeCells(`${groupLetter}${firstRow}:${groupLetter}${lastRow}`);
        sheet.getRange(`${groupLetter}${firstRow}`).values = [[schedule.rows[start].group]];
        applyStyle(sheet.getRange(`${groupLetter}${firstRow}:${groupLetter}${lastRow}`), {
          fill: groupFill(schedule.rows[start].group),
          bold: true,
          size: 11,
          borders: { preset: "outside", style: "medium", color: "#333333" },
        });
      }
      start = end + 1;
    }
  }

  sheet.freezePanes.freezeRows(layout.bodyStartRow);
  sheet.freezePanes.freezeColumns(layout.dayStartColumn);
}


function validateScheduleCellValues(sheet, schedule, layout) {
  for (let rowIndex = 0; rowIndex < schedule.rows.length; rowIndex += 1) {
    const person = schedule.rows[rowIndex];
    const row = layout.bodyStartRow + rowIndex;
    for (let dayIndex = 0; dayIndex < schedule.day_count; dayIndex += 1) {
      const column = layout.dayStartColumn + dayIndex;
      const actual = sheet.getRangeByIndexes(row, column, 1, 1).values[0][0];
      const expected = person.codes_display[dayIndex];
      if (actual !== expected) {
        throw new Error(
          `${schedule.schedule_id} code mismatch at row ${rowIndex + 1}, day ${dayIndex + 1}: ${actual} != ${expected}`,
        );
      }
    }
  }
}


function makeCellAnnotations(schedule, layout, imageWidth, imageHeight) {
  const totalWidth = layout.columnWidths.reduce((sum, value) => sum + value, 0);
  const totalHeight = layout.rowHeights.reduce((sum, value) => sum + value, 0);
  const xScale = imageWidth / totalWidth;
  const yScale = imageHeight / totalHeight;
  const xEdges = [0];
  const yEdges = [0];
  for (const width of layout.columnWidths) xEdges.push(xEdges.at(-1) + width);
  for (const height of layout.rowHeights) yEdges.push(yEdges.at(-1) + height);
  const annotations = [];

  function cellGeometry(row, column, rowSpan = 1, columnSpan = 1) {
    const left = Math.round(xEdges[column] * xScale);
    const top = Math.round(yEdges[row] * yScale);
    const right = Math.round(xEdges[column + columnSpan] * xScale);
    const bottom = Math.round(yEdges[row + rowSpan] * yScale);
    return {
      bbox: [left, top, right, bottom],
      polygon: [[left, top], [right, top], [right, bottom], [left, bottom]],
    };
  }

  // artifact-tool exposes the same explicit row/column layout that is rendered.
  // Text is center aligned in these sheets, so derive a conservative text layout
  // quadrilateral from that layout.  The training validator can replace this with
  // a glyph-mask/difference polygon when the rendered glyph check exceeds 2 px.
  function textGeometry(row, column, text, fontSize, rowSpan = 1, columnSpan = 1, align = "center") {
    const { bbox } = cellGeometry(row, column, rowSpan, columnSpan);
    const characters = [...String(text ?? "")];
    const hangul = characters.filter((char) => /[\u3131-\u318E\uAC00-\uD7A3]/u.test(char)).length;
    const ascii = characters.length - hangul;
    const scale = Math.min(xScale, yScale);
    const estimatedWidth = Math.max(2, (hangul * fontSize + ascii * fontSize * 0.62) * scale);
    const estimatedHeight = Math.max(2, fontSize * 1.25 * scale);
    const padding = Math.max(2, estimatedHeight * 0.05);
    const availableWidth = Math.max(2, bbox[2] - bbox[0] - 4);
    const availableHeight = Math.max(2, bbox[3] - bbox[1] - 4);
    const width = Math.min(availableWidth, estimatedWidth + 2 * padding);
    const height = Math.min(availableHeight, estimatedHeight + 2 * padding);
    const centerY = (bbox[1] + bbox[3]) / 2;
    let left;
    if (align === "left") left = bbox[0] + Math.min(6 * xScale, availableWidth - width);
    else left = (bbox[0] + bbox[2] - width) / 2;
    const top = centerY - height / 2;
    const right = left + width;
    const bottom = top + height;
    return [
      [Math.round(left), Math.round(top)],
      [Math.round(right), Math.round(top)],
      [Math.round(right), Math.round(bottom)],
      [Math.round(left), Math.round(bottom)],
    ];
  }

  const trainingObjects = [];
  function pushObject({
    objectType, displayText, row, column, rowSpan = 1, columnSpan = 1,
    fontSize = 9, align = "center", rowId = null, rowIndex = null,
    day = null, canonicalCode = null,
  }) {
    if (displayText === null || displayText === undefined || String(displayText) === "") return;
    const geometry = cellGeometry(row, column, rowSpan, columnSpan);
    trainingObjects.push({
      schedule_id: schedule.schedule_id,
      template_id: schedule.template_id,
      object_type: objectType,
      display_text: String(displayText).normalize("NFC"),
      canonical_code: canonicalCode,
      row_id: rowId,
      row_index: rowIndex,
      day,
      bbox_px: geometry.bbox,
      cell_polygon: geometry.polygon,
      text_polygon: textGeometry(row, column, displayText, fontSize, rowSpan, columnSpan, align),
      text_polygon_source: "layout_bounds",
      visibility: 1.0,
      ignore: false,
    });
  }

  const pageBadgeColumns = schedule.page_label ? 2 : 0;
  const titleColumns = layout.totalColumns - pageBadgeColumns;
  pushObject({ objectType: "title", displayText: schedule.title, row: 0, column: 0, columnSpan: titleColumns, fontSize: 15 });
  if (pageBadgeColumns) {
    pushObject({ objectType: "page_label", displayText: schedule.page_label, row: 0, column: titleColumns, columnSpan: pageBadgeColumns, fontSize: 9 });
  }
  if (layout.noteRow) {
    pushObject({
      objectType: "notice",
      displayText: "공지사항: 합성 데이터 / 실제 직원 정보 아님 / 모든 코드는 학습용 무작위 생성",
      row: 1,
      column: 0,
      columnSpan: layout.totalColumns,
      fontSize: 9,
      align: "left",
    });
  }
  const headerRow = layout.headerStartRow;
  if (layout.groupColumn) {
    pushObject({ objectType: "group_header", displayText: schedule.template_id === "compact_summary" ? "병동" : "구분", row: headerRow, column: layout.groupColumnIndex, rowSpan: layout.headerRows, fontSize: 10 });
    pushObject({ objectType: "name_header", displayText: "성명", row: headerRow, column: layout.nameColumn, rowSpan: layout.headerRows, fontSize: 10 });
  } else {
    pushObject({ objectType: "name_header", displayText: "성명", row: headerRow, column: 0, rowSpan: layout.headerRows, fontSize: 10 });
  }
  const annotationDateRow = headerRow + (layout.headerRows === 3 ? 1 : 0);
  if (layout.headerRows === 3) {
    pushObject({ objectType: "month_header", displayText: `${schedule.year}년 ${schedule.month}월 근무 일정`, row: headerRow, column: layout.dayStartColumn, columnSpan: schedule.day_count, fontSize: 10 });
  }
  schedule.weekdays.forEach((weekday, dayIndex) => {
    const column = layout.dayStartColumn + dayIndex;
    pushObject({ objectType: "date_header", displayText: dayIndex + 1, row: annotationDateRow, column, fontSize: 9, day: dayIndex + 1 });
    pushObject({ objectType: "weekday_header", displayText: weekday, row: annotationDateRow + 1, column, fontSize: 9, day: dayIndex + 1 });
  });

  layout.summaries.forEach((label, index) => {
    pushObject({ objectType: "summary_header", displayText: label, row: headerRow, column: layout.summaryStartColumn + index, rowSpan: layout.headerRows, fontSize: 9 });
  });

  schedule.rows.forEach((person, rowIndex) => {
    const sheetRowIndex = layout.bodyStartRow + rowIndex;
    const excelRow = sheetRowIndex + 1;
    const nameColumn = layout.nameColumn;
    person.excel_row = excelRow;
    person.name_cell = `${excelColumnName(nameColumn + 1)}${excelRow}`;
    const nameBox = [
      Math.round(xEdges[nameColumn] * xScale),
      Math.round(yEdges[sheetRowIndex] * yScale),
      Math.round(xEdges[nameColumn + 1] * xScale),
      Math.round(yEdges[sheetRowIndex + 1] * yScale),
    ];
    const nameGeometry = cellGeometry(sheetRowIndex, nameColumn);
    pushObject({
      objectType: "name", displayText: person.name, row: sheetRowIndex, column: nameColumn,
      fontSize: 10, rowId: person.row_id, rowIndex: rowIndex + 1,
    });
    if (layout.groupColumn && (rowIndex === 0 || schedule.rows[rowIndex - 1].group !== person.group)) {
      let groupRowSpan = 1;
      while (rowIndex + groupRowSpan < schedule.rows.length && schedule.rows[rowIndex + groupRowSpan].group === person.group) groupRowSpan += 1;
      pushObject({
        objectType: "group", displayText: person.group, row: sheetRowIndex, column: layout.groupColumnIndex,
        rowSpan: groupRowSpan, fontSize: 10, rowId: person.row_id, rowIndex: rowIndex + 1,
      });
    }
    person.codes_display.forEach((displayCode, dayIndex) => {
      const column = layout.dayStartColumn + dayIndex;
      const geometry = cellGeometry(sheetRowIndex, column);
      const textPolygon = textGeometry(sheetRowIndex, column, displayCode, codeFontSize(displayCode));
      annotations.push({
        schedule_id: schedule.schedule_id,
        template_id: schedule.template_id,
        row_id: person.row_id,
        row_index: rowIndex + 1,
        name: person.name,
        surname: person.surname,
        surname_rank: person.surname_rank,
        surname_population: person.surname_population,
        surname_hanja_variants: person.surname_hanja_variants,
        birth_year: person.birth_year,
        gender: person.gender,
        group: person.group,
        day: dayIndex + 1,
        date: `${String(schedule.year).padStart(4, "0")}-${String(schedule.month).padStart(2, "0")}-${String(dayIndex + 1).padStart(2, "0")}`,
        canonical_code: person.codes_canonical[dayIndex],
        display_code: displayCode,
        display_text: String(displayCode).normalize("NFC"),
        object_type: "shift_code",
        bbox_px: geometry.bbox,
        cell_polygon: geometry.polygon,
        text_polygon: textPolygon,
        text_polygon_source: "layout_bounds",
        visibility: 1.0,
        ignore: false,
        name_bbox_px: nameBox,
        name_cell_polygon: nameGeometry.polygon,
      });
      pushObject({
        objectType: "shift_code", displayText: displayCode, row: sheetRowIndex, column,
        fontSize: codeFontSize(displayCode), rowId: person.row_id, rowIndex: rowIndex + 1,
        day: dayIndex + 1, canonicalCode: person.codes_canonical[dayIndex],
      });
    });

    if (layout.summaries.length > 0) {
      const normalized = person.codes_canonical.map((code) => String(code).toUpperCase());
      const counts = {
        D: normalized.filter((code) => code === "D").length,
        E: normalized.filter((code) => code === "E").length,
        N: normalized.filter((code) => code === "N").length,
        OFF: normalized.filter((code) => ["OFF", "O", "F", "OF"].includes(code)).length,
        연차: person.codes_canonical.filter((code) => ["연", "연차", "연가", "AL", "A/L"].includes(code)).length,
      };
      layout.summaries.forEach((label, index) => {
        pushObject({
          objectType: "summary_value", displayText: counts[label] ?? 0,
          row: sheetRowIndex, column: layout.summaryStartColumn + index,
          fontSize: 9, rowId: person.row_id, rowIndex: rowIndex + 1,
        });
      });
    }
  });
  return { annotations, trainingObjects };
}


function configureTextMaskSheet(sheet, schedule, layout) {
  TEXT_MASK_MODE = true;
  try {
    configureScheduleSheet(sheet, schedule, layout);
  } finally {
    TEXT_MASK_MODE = false;
  }
}


function addTableSheet(workbook, name, headers, rows, options = {}) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.getRangeByIndexes(0, 0, 1, headers.length).values = [headers];
  if (rows.length > 0) sheet.getRangeByIndexes(1, 0, rows.length, headers.length).values = rows;
  applyStyle(sheet.getRangeByIndexes(0, 0, 1, headers.length), {
    fill: options.headerFill ?? "#1F4E78",
    bold: true,
    color: "#FFFFFF",
    size: 10,
  });
  if (rows.length > 0) {
    applyStyle(sheet.getRangeByIndexes(1, 0, rows.length, headers.length), {
      fill: "#FFFFFF",
      size: 9,
      horizontalAlignment: options.bodyAlignment ?? "left",
    });
  }
  sheet.freezePanes.freezeRows(1);
  headers.forEach((_header, index) => setColumnWidth(sheet, index, options.columnWidths?.[index] ?? 140));
  return sheet;
}


function populateReferenceSheets(workbook, payload) {
  const readme = workbook.worksheets.add("README");
  readme.showGridLines = false;
  readme.mergeCells("A1:H1");
  readme.getRange("A1").values = [["합성 간호사 근무표 데이터셋"]];
  applyStyle(readme.getRange("A1:H1"), { fill: "#1F4E78", bold: true, color: "#FFFFFF", size: 16 });
  const notes = [
    ["항목", "내용"],
    ["목적", "근무표 OCR 및 표 구조 인식 학습용 합성 데이터"],
    ["개인정보", "실제 직원 정보가 아닌 합성 이름과 무작위 근무 코드"],
    ["PNG 생성", "각 근무표 시트의 전체 사용 범위를 2배 해상도로 직접 렌더링"],
    ["엑셀", "통합 문서를 항상 보관하며 근무표·사전·정답지·출처 시트를 포함"],
    ["정답지", "ground_truth_rows 및 ground_truth_cells 시트"],
    ["이미지 좌표", "bbox_px는 최종 PNG 기준 [x1,y1,x2,y2]"],
  ];
  readme.getRangeByIndexes(2, 0, notes.length, 2).values = notes;
  applyStyle(readme.getRangeByIndexes(2, 0, 1, 2), { fill: "#D9EAF7", bold: true });
  applyStyle(readme.getRangeByIndexes(3, 0, notes.length - 1, 2), { fill: "#FFFFFF", horizontalAlignment: "left", wrapText: true });
  setColumnWidth(readme, 0, 150);
  setColumnWidth(readme, 1, 620);

  const codeRows = [];
  for (const [group, codes] of Object.entries(payload.shift_code_groups)) {
    for (const code of codes) codeRows.push([group, code, /[A-Za-z]/.test(code)]);
  }
  addTableSheet(workbook, "code_dictionary", ["group", "canonical_code", "english_case_mutable"], codeRows, {
    headerFill: "#385723",
    columnWidths: [130, 170, 180],
    bodyAlignment: "center",
  });

  addTableSheet(
    workbook,
    "names_by_birth_year",
    ["birth_year", "gender", "rank", "given_name", "source_year", "source_method", "source_url"],
    payload.name_entries.map((entry) => [entry.birth_year, entry.gender, entry.rank, entry.given_name, entry.source_year, entry.source_method, entry.source_url]),
    { headerFill: "#8064A2", columnWidths: [100, 90, 80, 110, 100, 190, 420] },
  );

  const surnameByText = Object.fromEntries(payload.surname_pool.map((item) => [item.surname, item]));
  addTableSheet(
    workbook,
    "surname_dictionary",
    ["rank", "surname", "hanja", "population", "source_year", "source_method", "source_url", "aggregated_population", "aggregated_best_rank", "hanja_variants", "sampling_weight_power_0_75"],
    payload.surname_entries.map((entry) => {
      const aggregate = surnameByText[entry.surname];
      return [entry.rank, entry.surname, entry.hanja, entry.population, entry.source_year, entry.source_method, entry.source_url, aggregate.population, aggregate.best_rank, aggregate.hanja_variants, Number((aggregate.population ** 0.75).toFixed(6))];
    }),
    { headerFill: "#7F6000", columnWidths: [70, 80, 90, 120, 100, 190, 420, 160, 150, 170, 190] },
  );

  addTableSheet(
    workbook,
    "sources",
    ["source_id", "purpose", "url_or_reference", "note"],
    payload.sources.map((source) => [source.source_id, source.purpose, source.url_or_reference, source.note]),
    { headerFill: "#5B9BD5", columnWidths: [150, 260, 460, 420] },
  );
}


function populateGroundTruthSheets(workbook, schedules) {
  const manifestRows = schedules.map((schedule) => [
    schedule.schedule_id,
    schedule.template_id,
    schedule.page_number,
    schedule.page_count,
    schedule.page_label,
    schedule.year,
    schedule.month,
    schedule.day_count,
    schedule.rows.length,
    schedule.sheet_name,
    schedule.clean_image_path,
    schedule.image_width,
    schedule.image_height,
  ]);
  addTableSheet(workbook, "manifest", ["schedule_id", "template_id", "page_number", "page_count", "page_label", "year", "month", "day_count", "people_count", "sheet_name", "clean_image_path", "image_width", "image_height"], manifestRows, {
    headerFill: "#1F4E78",
    columnWidths: [140, 150, 100, 100, 140, 80, 80, 90, 110, 140, 320, 110, 110],
  });

  const rowRows = [];
  const cellRows = [];
  const objectRows = [];
  for (const schedule of schedules) {
    for (let rowIndex = 0; rowIndex < schedule.rows.length; rowIndex += 1) {
      const person = schedule.rows[rowIndex];
      rowRows.push([
        schedule.schedule_id, schedule.template_id, schedule.page_number, schedule.page_count,
        schedule.page_label, schedule.sheet_name, person.row_id, rowIndex + 1,
        person.excel_row, person.group, person.name, person.surname, person.surname_rank,
        person.surname_population, person.surname_hanja_variants, person.surname_source_method,
        person.surname_source_url, person.given_name, person.birth_year, person.gender,
        schedule.day_count, JSON.stringify(person.codes_canonical), JSON.stringify(person.codes_display),
        person.codes_canonical.join("|"), person.codes_display.join("|"), person.name_cell,
      ]);
    }
    for (const annotation of schedule.cell_annotations) {
      const person = schedule.rows.find((row) => row.row_id === annotation.row_id);
      const layout = schedule._layout;
      const column = layout.dayStartColumn + annotation.day;
      const excelCell = `${excelColumnName(column)}${person.excel_row}`;
      cellRows.push([
        annotation.schedule_id, annotation.template_id, schedule.page_number, schedule.page_count,
        schedule.page_label, annotation.row_id, annotation.row_index,
        annotation.name, annotation.surname, annotation.surname_rank, annotation.surname_population,
        annotation.birth_year, annotation.gender, annotation.group, annotation.day, annotation.date,
        annotation.canonical_code, annotation.display_code, annotation.display_text, annotation.object_type, excelCell,
        JSON.stringify(annotation.bbox_px), JSON.stringify(annotation.cell_polygon), JSON.stringify(annotation.text_polygon),
        annotation.text_polygon_source, JSON.stringify(annotation.name_bbox_px), JSON.stringify(annotation.name_cell_polygon), schedule.clean_image_path,
      ]);
    }
    for (const annotation of schedule.training_objects) {
      objectRows.push([
        annotation.schedule_id, annotation.template_id, annotation.object_type,
        annotation.display_text, annotation.canonical_code, annotation.row_id,
        annotation.row_index, annotation.day, JSON.stringify(annotation.bbox_px),
        JSON.stringify(annotation.cell_polygon), JSON.stringify(annotation.text_polygon),
        annotation.text_polygon_source, annotation.visibility, annotation.ignore,
        schedule.clean_image_path, schedule.image_width, schedule.image_height,
      ]);
    }
  }

  addTableSheet(workbook, "ground_truth_rows", ["schedule_id", "template_id", "page_number", "page_count", "page_label", "sheet_name", "row_id", "row_index", "excel_row", "group", "name", "surname", "surname_rank", "surname_population", "surname_hanja_variants", "surname_source_method", "surname_source_url", "given_name", "birth_year", "gender", "day_count", "codes_canonical_json", "codes_display_json", "codes_canonical_joined", "codes_display_joined", "name_cell"], rowRows, {
    headerFill: "#C65911",
    columnWidths: Array(26).fill(150),
  });
  addTableSheet(workbook, "ground_truth_cells", ["schedule_id", "template_id", "page_number", "page_count", "page_label", "row_id", "row_index", "name", "surname", "surname_rank", "surname_population", "birth_year", "gender", "group", "day", "date", "canonical_code", "display_code", "display_text", "object_type", "excel_cell", "bbox_px_json", "cell_polygon_json", "text_polygon_json", "text_polygon_source", "name_bbox_px_json", "name_cell_polygon_json", "image_path"], cellRows, {
    headerFill: "#BF9000",
    columnWidths: Array(28).fill(150),
  });
  addTableSheet(workbook, "ground_truth_objects", ["schedule_id", "template_id", "object_type", "display_text", "canonical_code", "row_id", "row_index", "day", "bbox_px_json", "cell_polygon_json", "text_polygon_json", "text_polygon_source", "visibility", "ignore", "image_path", "image_width", "image_height"], objectRows, {
    headerFill: "#548235",
    columnWidths: Array(17).fill(150),
  });
}


async function verifyWorkbook(xlsxPath, outputDir, sheetNames) {
  const input = await FileBlob.load(xlsxPath);
  const workbook = await SpreadsheetFile.importXlsx(input);
  await fs.mkdir(outputDir, { recursive: true });
  const sheetSpecs = sheetNames.map((spec) => {
    const separator = spec.indexOf("=");
    return separator < 0
      ? { sheetName: spec, range: "A1:H20" }
      : { sheetName: spec.slice(0, separator), range: spec.slice(separator + 1) };
  });
  for (const { sheetName, range } of sheetSpecs) {
    const preview = await workbook.render({
      sheetName,
      range,
      scale: 1,
      format: "png",
      headers: false,
    });
    const bytes = new Uint8Array(await preview.arrayBuffer());
    await fs.writeFile(path.join(outputDir, `${sheetName}.png`), bytes);
  }
  const keyRange = await workbook.inspect({
    kind: "table",
    sheetId: sheetSpecs[0].sheetName,
    range: "A1:B9",
    include: "values,formulas",
    tableMaxRows: 12,
    tableMaxCols: 4,
    maxChars: 3000,
  });
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "verification formula error scan",
  });
  console.log(keyRange.ndjson);
  console.log(errors.ndjson);
}


async function main() {
  if (process.argv[2] === "--verify") {
    const [, , , xlsxPath, outputDir, ...sheetNames] = process.argv;
    if (!xlsxPath || !outputDir || sheetNames.length === 0) {
      throw new Error("usage: node render_workbook.mjs --verify <xlsx> <output-dir> <sheet[=range]>...");
    }
    await verifyWorkbook(xlsxPath, outputDir, sheetNames);
    return;
  }
  const [payloadPath, resultPath] = process.argv.slice(2);
  if (!payloadPath || !resultPath) throw new Error("usage: node render_workbook.mjs <payload.json> <result.json>");
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  const exportWorkbook = payload.export_workbook !== false;
  await fs.mkdir(payload.output_dir, { recursive: true });
  await fs.mkdir(path.join(payload.output_dir, "images"), { recursive: true });

  const workbook = Workbook.create();
  if (exportWorkbook) populateReferenceSheets(workbook, payload);

  for (const schedule of payload.schedules) {
    const layout = scheduleLayout(schedule);
    schedule._layout = layout;
    const sheet = workbook.worksheets.add(schedule.sheet_name);
    configureScheduleSheet(sheet, schedule, layout);
    validateScheduleCellValues(sheet, schedule, layout);
    const imageBlob = await workbook.render({
      sheetName: schedule.sheet_name,
      range: layout.usedRange,
      scale: RENDER_SCALE,
      format: "png",
      headers: false,
    });
    const imageBytes = new Uint8Array(await imageBlob.arrayBuffer());
    const dimensions = pngDimensions(imageBytes);
    const imageName = `${schedule.schedule_id}_${schedule.template_id}.png`;
    const imagePath = path.join(payload.output_dir, "images", imageName);
    await fs.writeFile(imagePath, imageBytes);
    schedule.clean_image_path = path.posix.join("images", imageName);
    schedule.image_width = dimensions.width;
    schedule.image_height = dimensions.height;
    const generatedAnnotations = makeCellAnnotations(
      schedule,
      layout,
      dimensions.width,
      dimensions.height,
    );
    schedule.cell_annotations = generatedAnnotations.annotations;
    schedule.training_objects = generatedAnnotations.trainingObjects;

    const maskWorkbook = Workbook.create();
    const maskSheet = maskWorkbook.worksheets.add("text_mask");
    configureTextMaskSheet(maskSheet, schedule, layout);
    const maskBlob = await maskWorkbook.render({
      sheetName: "text_mask",
      range: layout.usedRange,
      scale: RENDER_SCALE,
      format: "png",
      headers: false,
    });
    const maskBytes = new Uint8Array(await maskBlob.arrayBuffer());
    const maskDirectory = path.join(payload.output_dir, ".glyph_masks");
    await fs.mkdir(maskDirectory, { recursive: true });
    const maskName = `${schedule.schedule_id}_${schedule.template_id}_mask.png`;
    await fs.writeFile(path.join(maskDirectory, maskName), maskBytes);
    schedule.glyph_mask_path = path.posix.join(".glyph_masks", maskName);
  }

  if (exportWorkbook) populateGroundTruthSheets(workbook, payload.schedules);
  const formulaErrors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "final formula error scan",
  });
  let xlsxPath = null;
  if (exportWorkbook) {
    const xlsx = await SpreadsheetFile.exportXlsx(workbook);
    xlsxPath = path.join(payload.output_dir, payload.workbook_name ?? "synthetic_shift_dataset.xlsx");
    await xlsx.save(xlsxPath);
    await fs.rm(`${xlsxPath}.inspect.ndjson`, { force: true });
  }

  const result = {
    render_scale: RENDER_SCALE,
    export_workbook: exportWorkbook,
    xlsx_path: xlsxPath,
    formula_error_scan: formulaErrors.ndjson,
    schedules: payload.schedules.map((schedule) => ({
      schedule_id: schedule.schedule_id,
      page_number: schedule.page_number,
      page_count: schedule.page_count,
      page_label: schedule.page_label,
      clean_image_path: schedule.clean_image_path,
      image_width: schedule.image_width,
      image_height: schedule.image_height,
      rows: schedule.rows.map((person) => ({
        row_id: person.row_id,
        excel_row: person.excel_row,
        name_cell: person.name_cell,
      })),
      cell_annotations: schedule.cell_annotations,
      training_objects: schedule.training_objects,
      glyph_mask_path: schedule.glyph_mask_path,
    })),
  };
  await fs.writeFile(resultPath, JSON.stringify(result), "utf8");
}


await main();

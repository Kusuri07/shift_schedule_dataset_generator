import fs from "node:fs/promises";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


function styleHeader(range, fill) {
  range.format = {
    fill,
    font: { name: "Malgun Gothic", bold: true, color: "#FFFFFF", size: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "medium", color: fill },
  };
}


function writeRowsInChunks(sheet, startRow, rows, columnCount, chunkSize = 1000) {
  for (let start = 0; start < rows.length; start += chunkSize) {
    const chunk = rows.slice(start, start + chunkSize);
    sheet.getRangeByIndexes(startRow + start, 0, chunk.length, columnCount).values = chunk;
  }
}


async function main() {
  const [outputDir, outputPath] = process.argv.slice(2);
  if (!outputDir || !outputPath) {
    throw new Error("usage: node build_print_index.mjs <print-output-dir> <output-xlsx>");
  }

  const printManifest = JSON.parse(
    await fs.readFile(`${outputDir}/print_manifest.json`, "utf8"),
  );
  const annotationManifest = JSON.parse(
    await fs.readFile(`${outputDir}/annotations/manifest.json`, "utf8"),
  );
  const rowRecords = (await fs.readFile(`${outputDir}/annotations/rows.jsonl`, "utf8"))
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));

  const workbook = Workbook.create();
  const guide = workbook.worksheets.add("안내");
  const index = workbook.worksheets.add("인쇄 인덱스");
  const rowTruth = workbook.worksheets.add("행 정답");

  guide.showGridLines = false;
  guide.mergeCells("A1:J1");
  guide.getRange("A1").values = [[
    `합성 병동 근무표 ${printManifest.schedule_count.toLocaleString("ko-KR")}장 - 정답 확인 인덱스`,
  ]];
  guide.getRange("A1:J1").format = {
    fill: "#1F4E78",
    font: { name: "Malgun Gothic", bold: true, color: "#FFFFFF", size: 18 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    rowHeight: 36,
  };
  const pdfCount = new Set(printManifest.records.map((record) => record.pdf_file)).size;
  guide.getRange("A3:B9").values = [
    ["항목", "값"],
    ["근무표 수", null],
    ["PDF 파일 수", pdfCount],
    ["행 정답 수", null],
    ["셀 정답 수", annotationManifest.cell_answer_count],
    ["첫 페이지", printManifest.first_page],
    ["마지막 페이지", printManifest.last_page],
  ];
  guide.getRange("B4").formulas = [[
    `=COUNTA('인쇄 인덱스'!$A$2:$A$${printManifest.schedule_count + 1})`,
  ]];
  guide.getRange("B6").formulas = [[
    `=COUNTA('행 정답'!$A$2:$A$${rowRecords.length + 1})`,
  ]];
  guide.getRange("A3:B3").format = {
    fill: "#D9EAF7",
    font: { name: "Malgun Gothic", bold: true, color: "#17365D" },
    horizontalAlignment: "center",
  };
  guide.getRange("A3:B9").format.borders = {
    preset: "all",
    style: "thin",
    color: "#CBD5E1",
  };
  guide.getRange("A4:A9").format.font = { name: "Malgun Gothic", bold: true };
  guide.getRange("B4:B9").format.numberFormat = "#,##0";

  guide.mergeCells("A11:J11");
  guide.getRange("A11").values = [["정답 파일 사용법"]];
  guide.getRange("A11:J11").format = {
    fill: "#F8B800",
    font: { name: "Malgun Gothic", bold: true, color: "#111827", size: 13 },
  };
  guide.getRange("A12:C18").values = [
    ["파일", "용도", "주요 연결 키"],
    ["행 정답 시트", "직원 행별 이름·속성·월 전체 근무 코드", "schedule_id, row_id"],
    ["annotations/rows.csv", "행 정답 원본(CSV)", "schedule_id, row_id"],
    ["annotations/rows.jsonl", "행 정답 원본(JSONL)", "schedule_id, row_id"],
    ["annotations/cells.csv", "날짜별 코드와 PNG 좌표 정답", "schedule_id, row_id"],
    ["annotations/cells.jsonl", "날짜별 정답 원본(JSONL)", "schedule_id, row_id"],
    ["canonical_code / display_code", "학습 정답 / 이미지 표시값", "row_id + day"],
  ];
  guide.getRange("A12:C12").format = {
    fill: "#D9EAF7",
    font: { name: "Malgun Gothic", bold: true, color: "#17365D" },
    horizontalAlignment: "center",
  };
  guide.getRange("A13:C18").format = {
    fill: "#F8FAFC",
    font: { name: "Malgun Gothic", color: "#1F2937", size: 10 },
    verticalAlignment: "center",
  };
  guide.mergeCells("A20:J20");
  guide.getRange("A20").values = [["중요"]];
  guide.getRange("A20:J20").format = {
    fill: "#C65911",
    font: { name: "Malgun Gothic", bold: true, color: "#FFFFFF", size: 12 },
  };
  guide.mergeCells("A21:J22");
  guide.getRange("A21").values = [[
    `셀 정답 ${annotationManifest.cell_answer_count.toLocaleString("ko-KR")}건은 열람 성능을 위해 CSV/JSONL로 보존했습니다. ` +
    "bbox_px는 [왼쪽, 위, 오른쪽, 아래] 순서의 PNG 픽셀 좌표입니다.",
  ]];
  guide.getRange("A21:J22").format = {
    fill: "#FFF2CC",
    font: { name: "Malgun Gothic", color: "#7C2D12", size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  guide.getRange("A:A").format.columnWidth = 31;
  guide.getRange("B:B").format.columnWidth = 34;
  guide.getRange("C:C").format.columnWidth = 28;
  guide.getRange("D:J").format.columnWidth = 12;
  guide.getRange("3:20").format.rowHeight = 24;
  guide.getRange("21:22").format.rowHeight = 32;

  const indexHeaders = [
    "page_number", "page_count", "page_label", "schedule_id", "template_id",
    "year", "month", "people_count", "image_file", "pdf_file", "pdf_page",
  ];
  index.getRangeByIndexes(0, 0, 1, indexHeaders.length).values = [indexHeaders];
  index.getRangeByIndexes(1, 0, printManifest.records.length, indexHeaders.length).values =
    printManifest.records.map((record) => indexHeaders.map((field) => record[field]));
  index.showGridLines = false;
  index.freezePanes.freezeRows(1);
  styleHeader(index.getRange("A1:K1"), "#1F4E78");
  index.getRange("A:A").format.columnWidth = 12;
  index.getRange("B:B").format.columnWidth = 12;
  index.getRange("C:C").format.columnWidth = 18;
  index.getRange("D:E").format.columnWidth = 22;
  index.getRange("F:H").format.columnWidth = 11;
  index.getRange("I:I").format.columnWidth = 48;
  index.getRange("J:J").format.columnWidth = 45;
  index.getRange("K:K").format.columnWidth = 11;
  const indexLastRow = printManifest.records.length + 1;
  const indexTable = index.tables.add(`A1:K${indexLastRow}`, true, "PrintIndexTable");
  indexTable.style = "TableStyleMedium2";
  indexTable.showFilterButton = true;

  const rowHeaders = [
    "schedule_id", "template_id", "page_number", "page_count", "page_label",
    "sheet_name", "row_id", "row_index", "excel_row", "group", "name", "surname",
    "surname_rank", "surname_population", "surname_hanja_variants",
    "surname_source_method", "surname_source_url", "given_name", "birth_year", "gender",
    "day_count", "codes_canonical", "codes_display", "codes_canonical_joined",
    "codes_display_joined", "name_cell", "image_path",
  ];
  rowTruth.getRangeByIndexes(0, 0, 1, rowHeaders.length).values = [rowHeaders];
  const rowValues = rowRecords.map((record) => rowHeaders.map((field) => {
    const value = record[field];
    return Array.isArray(value) ? JSON.stringify(value) : value;
  }));
  writeRowsInChunks(rowTruth, 1, rowValues, rowHeaders.length);
  rowTruth.showGridLines = false;
  rowTruth.freezePanes.freezeRows(1);
  rowTruth.freezePanes.freezeColumns(7);
  styleHeader(rowTruth.getRange("A1:AA1"), "#C65911");
  rowTruth.getRange("A:G").format.columnWidth = 18;
  rowTruth.getRange("H:N").format.columnWidth = 13;
  rowTruth.getRange("O:Q").format.columnWidth = 30;
  rowTruth.getRange("R:U").format.columnWidth = 13;
  rowTruth.getRange("V:Y").format.columnWidth = 42;
  rowTruth.getRange("Z:AA").format.columnWidth = 22;
  const rowLastRow = rowRecords.length + 1;
  const rowTable = rowTruth.tables.add(`A1:AA${rowLastRow}`, true, "RowTruthTable");
  rowTable.style = "TableStyleMedium9";
  rowTable.showFilterButton = true;

  for (const [sheetName, range] of [
    ["안내", "A1:J22"],
    ["인쇄 인덱스", `A1:K${Math.min(indexLastRow, 12)}`],
    ["행 정답", `A1:AA${Math.min(rowLastRow, 8)}`],
  ]) {
    const check = await workbook.inspect({
      kind: "table",
      range: `${sheetName}!${range}`,
      include: "values,formulas",
      tableMaxRows: 22,
      tableMaxCols: 27,
      maxChars: 8000,
    });
    console.log(check.ndjson);
  }
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "final formula error scan",
  });
  console.log(errors.ndjson);

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });
  console.log(`OUTPUT_XLSX=${outputPath}`);
}


await main();

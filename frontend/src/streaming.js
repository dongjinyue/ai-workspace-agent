export function createSseParser(onEvent) {
  let buffer = "";

  function parseBlock(block) {
    const lines = block.split(/\r?\n/);
    let event = "message";
    const dataLines = [];
    for (const line of lines) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return;
    const rawData = dataLines.join("\n");
    onEvent({ event, data: JSON.parse(rawData) });
  }

  return {
    push(chunk) {
      buffer += chunk;
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || "";
      blocks.filter((block) => block.trim()).forEach(parseBlock);
    },
    finish() {
      if (buffer.trim()) parseBlock(buffer);
      buffer = "";
    },
  };
}

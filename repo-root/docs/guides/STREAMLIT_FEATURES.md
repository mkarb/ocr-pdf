# Streamlit UI Features

## Enhanced PDF Ingestion with Progress Tracking

The Streamlit UI now includes comprehensive progress tracking and debugging capabilities for PDF ingestion.

### Features

#### 1. **Multi-Level Progress Bars**

- **Overall Progress**: Shows progress across all files being ingested
- **Page Progress**: Shows detailed page-by-page extraction progress for each file
- **Visual Indicators**: Uses emoji icons and color-coded messages for better UX

#### 2. **Debug Mode**

Enable detailed logging by checking the "Enable debug logging" checkbox:

- **Timestamped Events**: All extraction steps are logged with timestamps
- **Real-time Updates**: Debug feed updates as each page is processed
- **Performance Metrics**: Shows extraction speed (pages/second) and total time
- **Error Tracking**: Detailed error messages with context

#### 3. **Performance Metrics**

After ingestion, you'll see:
- Total processing time
- Pages per second throughput
- Document ID and page count
- File-specific statistics

### Debug Feed Information

The debug feed shows:
- File save operations
- Page count detection
- Vector extraction progress per page
- Database storage operations
- Error messages with full context
- Timing information

### Example Output

```
✅ Ingested drawing.pdf as `a1b2c3d4` (150 pages in 45.3s, 3.3 pages/sec)
```

### Debug Log Sample

```
[22:15:01] Saving drawing.pdf to disk...
[22:15:01] File saved to H:\repo-root\ocr-pdf\repo-root\uploads\drawing.pdf
[22:15:01] PDF has 150 pages
[22:15:01] Starting vector extraction...
[22:15:02] Extracted page 1/150 (0.7%)
[22:15:02] Extracted page 2/150 (1.3%)
...
[22:15:45] Extracted page 150/150 (100.0%)
[22:15:45] Vector extraction complete
[22:15:45] Storing in database...
[22:15:46] Database storage complete (total time: 45.32s)
```

## Usage

### Basic Ingestion

1. Upload one or more PDF files
2. Click "Ingest selected PDFs"
3. Watch the progress bars for real-time status

### Debug Mode

1. Check "Enable debug logging"
2. Upload and ingest files
3. Expand "Debug Feed" to see detailed logs
4. Monitor page-by-page extraction progress

### Performance Tips

- **Large PDFs**: The page progress bar shows exact page being processed
- **Multiple Files**: Overall progress shows which file is being processed
- **Errors**: Debug mode shows exact point of failure

## Technical Details

### Server Mode

The UI now uses `pdf_to_vectormap_server` which provides:
- Progress callback support
- Better performance metrics
- Parallel page processing
- Memory-efficient extraction

### Progress Callback

The progress callback fires after each page is extracted, updating:
- Page progress bar
- Debug feed (if enabled)
- Real-time statistics

## Troubleshooting

### If Progress Seems Stuck

Check the debug feed to see:
- Which page is being processed
- If there are any warnings
- System resource usage

### Performance Issues

The debug feed shows:
- Pages/second throughput
- Total extraction time
- Potential bottlenecks

Enable debug mode to identify slow pages or processing issues.

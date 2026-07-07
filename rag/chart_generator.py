"""
Chart generation module for the DataIntern RAG Engine.

Loads CRM datasets, applies filters and aggregation, and produces
interactive Plotly charts as serialisable JSON dictionaries.
"""

import json
import logging
import os

import pandas as pd
import plotly.express as px

logger = logging.getLogger(__name__)

# Supported chart types and their Plotly Express constructors.
_CHART_BUILDERS = {
    "bar": px.bar,
    "line": px.line,
    "pie": px.pie,
    "scatter": px.scatter,
    "histogram": px.histogram,
}


class ChartGenerator:
    """Generates Plotly charts from CRM data using parameters supplied by
    :class:`~rag.chart_detector.ChartDetector`.

    Attributes:
        data_dir: Absolute path to the directory containing CRM data files.
    """

    def __init__(self, data_dir: str) -> None:
        """Initialise the chart generator.

        Args:
            data_dir: Path to the directory that contains the CRM CSV, JSON,
                and XLSX data files.
        """
        self.data_dir = data_dir
        logger.info("ChartGenerator initialised with data_dir=%s", data_dir)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_dataset(self, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
        """Load a dataset by filename.

        Supports ``.csv``, ``.json``, and ``.xlsx`` files.  For JSON files
        whose top-level structure is a dict with a single list value
        (e.g. ``{"accounts": [...]}``) the nested list is automatically
        unwrapped.

        Args:
            filename: Name of the file to load (relative to ``data_dir``).
            sheet_name: Optional sheet name for XLSX files.  Defaults to the
                first sheet.

        Returns:
            A ``pandas.DataFrame`` with the loaded data.

        Raises:
            FileNotFoundError: If the file does not exist in ``data_dir``.
            ValueError: If the file format is not supported.
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        ext = os.path.splitext(filename)[1].lower()

        if ext == ".csv":
            df = pd.read_csv(filepath)
            logger.info("Loaded CSV '%s': %d rows × %d cols.", filename, *df.shape)
            return df

        if ext == ".json":
            df = self._load_json(filepath, filename)
            logger.info("Loaded JSON '%s': %d rows × %d cols.", filename, *df.shape)
            return df

        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath, sheet_name=sheet_name or 0)
            logger.info(
                "Loaded XLSX '%s' (sheet=%s): %d rows × %d cols.",
                filename,
                sheet_name or "first",
                *df.shape,
            )
            return df

        raise ValueError(f"Unsupported file format: '{ext}' (file: {filename})")

    # ------------------------------------------------------------------
    # Chart generation
    # ------------------------------------------------------------------

    def generate_chart(self, params: dict) -> dict:
        """Create a Plotly chart and return it as a JSON-serialisable dict.

        Steps:
        1. Load the dataset specified by ``params["dataset"]``.
        2. Apply column-value filters from ``params["filters"]``.
        3. Apply aggregation if ``params["aggregation"]`` is specified.
        4. Build the Plotly figure based on ``params["chart_type"]``.
        5. Return the figure as a parsed JSON dict.

        Args:
            params: A dict produced by
                :meth:`~rag.chart_detector.ChartDetector.detect_chart_params`.
                Expected keys: ``chart_type``, ``dataset``, ``x_column``,
                and optionally ``y_column``, ``color_column``, ``filters``,
                ``aggregation``, ``title``.

        Returns:
            A dict representation of the Plotly figure (``fig.to_json()``
            parsed back into a Python dict).

        Raises:
            ValueError: If the chart type is unsupported or required columns
                are missing from the dataset.
        """
        chart_type: str = params.get("chart_type", "bar")
        dataset: str = params["dataset"]
        x_col: str = params.get("x_column", "")
        y_col: str | None = params.get("y_column")
        color_col: str | None = params.get("color_column")
        filters: dict = params.get("filters", {})
        aggregation: str | None = params.get("aggregation")
        title: str = params.get("title", "Chart")

        if chart_type not in _CHART_BUILDERS:
            raise ValueError(
                f"Unsupported chart type '{chart_type}'. "
                f"Choose from {list(_CHART_BUILDERS)}."
            )

        # 1. Load
        sheet = params.get("sheet_name")
        df = self.load_dataset(dataset, sheet_name=sheet)

        # 2. Filter
        if filters:
            df = self._apply_filters(df, filters)

        # Validate columns exist.
        self._validate_columns(df, x_col, y_col, chart_type)

        # 3. Aggregate
        if aggregation and y_col and chart_type != "histogram":
            df = self._aggregate(df, x_col, y_col, aggregation, color_col)

        # 4. Build figure
        fig = self._build_figure(chart_type, df, x_col, y_col, color_col, title)

        # 5. Serialise
        figure_json: dict = json.loads(fig.to_json())
        logger.info("Generated '%s' chart: %s", chart_type, title)
        return figure_json

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(filepath: str, filename: str) -> pd.DataFrame:
        """Load a JSON file, handling nested single-key structures."""
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            # Unwrap single-key dict whose value is a list (e.g. {"accounts": [...]}).
            list_values = [v for v in data.values() if isinstance(v, list)]
            if len(list_values) == 1:
                return pd.DataFrame(list_values[0])
            # Fall back to a flat DataFrame from the dict.
            return pd.DataFrame([data])

        raise ValueError(f"Cannot convert JSON in '{filename}' to DataFrame.")

    @staticmethod
    def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """Apply column-value filters to a DataFrame.

        Each key-value pair in *filters* is treated as an equality filter.
        Filtering is case-insensitive for string columns.

        Args:
            df: The source DataFrame.
            filters: A mapping of ``{column_name: value}``.

        Returns:
            A filtered DataFrame.
        """
        for col, value in filters.items():
            if col not in df.columns:
                logger.warning("Filter column '%s' not in DataFrame; skipping.", col)
                continue
            if df[col].dtype == object:
                # Case-insensitive comparison for string columns.
                df = df[df[col].astype(str).str.lower() == str(value).lower()]
            else:
                df = df[df[col] == value]
            logger.debug("Filtered '%s' == '%s': %d rows remaining.", col, value, len(df))
        return df

    @staticmethod
    def _aggregate(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        aggregation: str,
        color_col: str | None = None,
    ) -> pd.DataFrame:
        """Group and aggregate the DataFrame.

        Args:
            df: The source DataFrame.
            x_col: Column to group by.
            y_col: Column to aggregate.
            aggregation: Aggregation function name (``sum``, ``count``,
                ``mean``, ``median``, ``min``, ``max``).
            color_col: Optional secondary grouping column.

        Returns:
            An aggregated DataFrame.
        """
        group_cols = [x_col]
        if color_col and color_col in df.columns:
            group_cols.append(color_col)

        valid_agg = {"sum", "count", "mean", "median", "min", "max"}
        if aggregation not in valid_agg:
            logger.warning(
                "Unknown aggregation '%s'; falling back to 'sum'.", aggregation
            )
            aggregation = "sum"

        df = df.groupby(group_cols, as_index=False).agg({y_col: aggregation})
        logger.debug(
            "Aggregated by %s (%s on %s): %d rows.",
            group_cols,
            aggregation,
            y_col,
            len(df),
        )
        return df

    @staticmethod
    def _validate_columns(
        df: pd.DataFrame,
        x_col: str,
        y_col: str | None,
        chart_type: str,
    ) -> None:
        """Ensure the required columns are present in the DataFrame."""
        if x_col and x_col not in df.columns:
            raise ValueError(
                f"x_column '{x_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        if y_col and y_col not in df.columns and chart_type != "histogram":
            raise ValueError(
                f"y_column '{y_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    @staticmethod
    def _build_figure(
        chart_type: str,
        df: pd.DataFrame,
        x_col: str,
        y_col: str | None,
        color_col: str | None,
        title: str,
    ):
        """Construct the Plotly Express figure."""
        # Resolve colour column to None if it's not in the DataFrame.
        if color_col and color_col not in df.columns:
            logger.warning("color_column '%s' not in data; ignoring.", color_col)
            color_col = None

        if chart_type == "histogram":
            return px.histogram(df, x=x_col, title=title)

        if chart_type == "pie":
            return px.pie(df, names=x_col, values=y_col, title=title)

        # bar / line / scatter all share the same signature.
        builder = _CHART_BUILDERS[chart_type]
        kwargs: dict = {"x": x_col, "y": y_col, "title": title}
        if color_col:
            kwargs["color"] = color_col
        return builder(df, **kwargs)

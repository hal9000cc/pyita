"""Metadata management for indicators."""
import importlib
import json
import re
from pathlib import Path

from .exceptions import PyTAExceptionMetadataParseError, PyTAExceptionMetadataError


def _parse_docstring(indicator_name, docstring):
    """Parse indicator docstring and extract metadata.
    
    Expected format:
    '''имя(параметры)

    Описание.

    Output series: серия1, серия2, серия3'''
    
    Args:
        indicator_name: Name of the indicator
        docstring: Module docstring
        
    Returns:
        dict: Dictionary with metadata:
            - name: Indicator name
            - signature: Full signature string
            - parameters: List of parameter names
            - output_series: List of output series names
            - description: Description string
            
    Raises:
        PyTAExceptionMetadataParseError: If parsing fails
    """
    if not docstring:
        raise PyTAExceptionMetadataParseError(
            indicator_name,
            "docstring is empty or missing"
        )
    
    lines = [line.strip() for line in docstring.strip().split('\n')]
    lines = [line for line in lines if line]
    
    if len(lines) < 3:
        raise PyTAExceptionMetadataParseError(
            indicator_name,
            f"docstring has {len(lines)} non-empty lines, expected at least 3"
        )
    
    signature_line = lines[0]
    if '(' not in signature_line or ')' not in signature_line:
        raise PyTAExceptionMetadataParseError(
            indicator_name,
            f"signature line does not contain function signature: {signature_line}"
        )
    
    name_from_signature = signature_line.split('(')[0].strip()
    if name_from_signature != indicator_name:
        raise PyTAExceptionMetadataParseError(
            indicator_name,
            f"indicator name mismatch: expected '{indicator_name}', got '{name_from_signature}'"
        )
    
    description_line = lines[1]
    if not description_line:
        raise PyTAExceptionMetadataParseError(
            indicator_name,
            "description line is empty"
        )
    
    output_series_line = None
    for line in lines[2:]:
        if line.startswith('Output series:'):
            output_series_line = line
            break
    
    if output_series_line is None:
        raise PyTAExceptionMetadataParseError(
            indicator_name,
            "missing 'Output series:' line"
        )
    
    output_series_text = output_series_line.replace('Output series:', '').strip()
    if output_series_text.endswith(')'):
        output_series_text = output_series_text[:-1]
    
    output_series = []
    for series_item in output_series_text.split(','):
        series_item = series_item.strip()
        if not series_item:
            continue
        
        # Parse format: "name (type)" or just "name"
        type_match = re.match(r'(\w+)\s*\(([^)]+)\)', series_item)
        if type_match:
            series_name = type_match.group(1)
            series_type = type_match.group(2).strip()
            # Normalize type: "as source" -> "as_source"
            if series_type == 'as source':
                series_type = 'as_source'
            elif series_type not in ('price', 'as_source'):
                series_type = 'none'
        else:
            series_name = series_item
            series_type = 'none'
        
        output_series.append({
            'name': series_name,
            'type': series_type
        })
    
    signature_match = re.match(r'(\w+)\((.*?)\)', signature_line)
    if not signature_match:
        raise PyTAExceptionMetadataParseError(
            indicator_name,
            f"cannot parse signature: {signature_line}"
        )
    
    params_text = signature_match.group(2)
    parameters = []
    if params_text.strip():
        for param in params_text.split(','):
            param = param.strip()
            if '=' in param:
                param_name = param.split('=')[0].strip()
            else:
                param_name = param.strip()
            if param_name:
                parameters.append(param_name)
    
    return {
        'name': indicator_name,
        'signature': signature_line,
        'parameters': parameters,
        'output_series': output_series,
        'description': description_line,
    }


def create_metadata():
    """Create metadata JSON file from all indicator modules.
    
    Scans indicators directory, imports each module, parses docstrings,
    and saves metadata to metadata.json file.
    
    Raises:
        PyTAExceptionMetadataParseError: If parsing fails for any indicator
    """
    indicators_dir = Path(__file__).parent / 'indicators'
    metadata_dict = {}
    
    for py_file in sorted(indicators_dir.glob('*.py')):
        if py_file.name == '__init__.py':
            continue
        
        indicator_name = py_file.stem
        
        try:
            module = importlib.import_module(f'pyita.indicators.{indicator_name}')
            docstring = module.__doc__
            
            metadata = _parse_docstring(indicator_name, docstring)
            metadata_dict[indicator_name] = metadata
            
        except Exception as e:
            if isinstance(e, PyTAExceptionMetadataParseError):
                raise
            raise PyTAExceptionMetadataParseError(
                indicator_name,
                f"error importing or parsing module: {str(e)}"
            ) from e
    
    metadata_file = Path(__file__).parent / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)


def metadata():
    """Get metadata for all indicators.
    
    Reads metadata from metadata.json file.
    
    Returns:
        dict: Dictionary mapping indicator names to their metadata.
              Each metadata dict contains:
              - name: Indicator name
              - signature: Full signature string
              - parameters: List of parameter names
              - output_series: List of output series names
              - description: Description string
              
    Raises:
        PyTAExceptionMetadataParseError: If metadata file is missing or invalid
    """
    metadata_file = Path(__file__).parent / 'metadata.json'
    
    if not metadata_file.exists():
        raise PyTAExceptionMetadataError(
            f"metadata file not found: {metadata_file}. Run create_metadata() first."
        )
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise PyTAExceptionMetadataError(
            f"invalid JSON in metadata file: {str(e)}"
        ) from e
    except Exception as e:
        raise PyTAExceptionMetadataError(
            f"error reading metadata file: {str(e)}"
        ) from e
    
    return metadata_dict


def list():
    """Get formatted list of all indicators in human-readable format.
    
    Returns:
        str: Formatted string with all indicators, their signatures,
             descriptions, and output series.
             
    Raises:
        PyTAExceptionMetadataError: If metadata file is missing or invalid
    """
    metadata_dict = metadata()
    
    lines = []
    for indicator_name in sorted(metadata_dict.keys()):
        meta = metadata_dict[indicator_name]
        lines.append(meta['signature'])
        lines.append(f"  {meta['description']}.")
        
        # Format output series with types in parentheses
        output_series_formatted = []
        for series in meta['output_series']:
            series_name = series['name']
            series_type = series['type']
            if series_type == 'none':
                output_series_formatted.append(series_name)
            else:
                # Convert 'as_source' back to 'as source' for display
                display_type = 'as source' if series_type == 'as_source' else series_type
                output_series_formatted.append(f"{series_name} ({display_type})")
        
        lines.append(f"  Output: {', '.join(output_series_formatted)}")
        lines.append('')
    
    return '\n'.join(lines)


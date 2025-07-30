import torch
import torch.nn as nn
import functools
from typing import Any, Dict, List, Optional

class DebugLogger:
    """Centralized debug logger with different verbosity levels"""
    
    def __init__(self, level='INFO', prefix='[DEBUG]'):
        self.level = level
        self.prefix = prefix
        self.levels = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}
        self.current_level = self.levels.get(level, 1)
    
    def debug(self, msg, force=False):
        if self.current_level <= 0 or force:
            print(f"{self.prefix}[DEBUG] {msg}")
    
    def info(self, msg, force=False):
        if self.current_level <= 1 or force:
            print(f"{self.prefix}[INFO] {msg}")
    
    def warning(self, msg, force=False):
        if self.current_level <= 2 or force:
            print(f"{self.prefix}[WARNING] {msg}")
    
    def error(self, msg, force=True):
        if self.current_level <= 3 or force:
            print(f"{self.prefix}[ERROR] {msg}")

# Global debug logger
debug_logger = DebugLogger(level='DEBUG', prefix='[PHI-SEG]')

def debug_tensor(tensor, name="tensor", logger=None, detailed=False):
    """Debug tensor properties"""
    if logger is None:
        logger = debug_logger
    
    if detailed:
        logger.info(f"{name}:")
        logger.info(f"  Shape: {tensor.shape}")
        logger.info(f"  Dtype: {tensor.dtype}")
        logger.info(f"  Device: {tensor.device}")
        logger.info(f"  Min: {tensor.min().item():.4f}")
        logger.info(f"  Max: {tensor.max().item():.4f}")
        logger.info(f"  Mean: {tensor.mean().item():.4f}")
        logger.info(f"  Std: {tensor.std().item():.4f}")
        logger.info(f"  Memory: {tensor.numel() * tensor.element_size() / 1024 / 1024:.2f} MB")
    else:
        logger.info(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    
    return tensor

def debug_module(module, name="module", logger=None):
    """Debug module properties"""
    if logger is None:
        logger = debug_logger
    
    logger.info(f"{name}: {type(module).__name__}")
    
    # Check for common attributes
    attrs_to_check = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
    for attr in attrs_to_check:
        if hasattr(module, attr):
            logger.info(f"  {attr}: {getattr(module, attr)}")
    
    return module

def debug_forward_hook(name="", logger=None, detailed=False):
    """Create a forward hook for debugging"""
    if logger is None:
        logger = debug_logger
    
    def hook(module, input, output):
        logger.info(f"=== Forward Hook: {name} ===")
        
        # Debug input
        if isinstance(input, (tuple, list)):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    debug_tensor(inp, f"input[{i}]", logger, detailed)
        elif isinstance(input, torch.Tensor):
            debug_tensor(input, "input", logger, detailed)
        
        # Debug output
        if isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    debug_tensor(out, f"output[{i}]", logger, detailed)
        elif isinstance(output, torch.Tensor):
            debug_tensor(output, "output", logger, detailed)
        
        logger.info(f"=== End Hook: {name} ===")
    
    return hook

def add_debug_hooks(model, hook_names=None, detailed=False):
    """Add debug hooks to model layers"""
    if hook_names is None:
        hook_names = {}
    
    hooks = []
    
    def add_hooks_recursive(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            hook_name = hook_names.get(full_name, full_name)
            
            # Add hook to this module
            hook = child.register_forward_hook(debug_forward_hook(hook_name, detailed=detailed))
            hooks.append(hook)
            
            # Recurse
            add_hooks_recursive(child, full_name)
    
    add_hooks_recursive(model)
    return hooks

def debug_model_structure(model, max_depth=3, logger=None):
    """Debug model structure recursively"""
    if logger is None:
        logger = debug_logger
    
    def print_structure(module, depth=0, prefix=""):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        logger.info(f"{indent}{prefix}: {type(module).__name__}")
        
        # Print module-specific info
        if hasattr(module, 'in_channels'):
            logger.info(f"{indent}  in_channels: {module.in_channels}")
        if hasattr(module, 'out_channels'):
            logger.info(f"{indent}  out_channels: {module.out_channels}")
        
        # Recurse through children
        for name, child in module.named_children():
            print_structure(child, depth + 1, name)
    
    logger.info("=== Model Structure ===")
    print_structure(model)

# Decorator for debugging functions
def debug_function(func_name=None, logger=None, log_args=True, log_return=True):
    """Decorator to add debug logging to functions"""
    if logger is None:
        logger = debug_logger
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            logger.info(f">>> Entering {name}")
            
            if log_args:
                logger.debug(f"    Args: {len(args)} positional")
                logger.debug(f"    Kwargs: {list(kwargs.keys())}")
            
            try:
                result = func(*args, **kwargs)
                if log_return:
                    if isinstance(result, torch.Tensor):
                        debug_tensor(result, f"{name}_output", logger)
                    elif isinstance(result, (tuple, list)):
                        logger.debug(f"    Returned {len(result)} items")
                
                logger.info(f"<<< Exiting {name}")
                return result
            
            except Exception as e:
                logger.error(f"!!! Error in {name}: {e}")
                raise
        
        return wrapper
    return decorator

# Context manager for temporary debug level
class DebugLevel:
    """Context manager to temporarily change debug level"""
    
    def __init__(self, level, logger=None):
        if logger is None:
            logger = debug_logger
        self.logger = logger
        self.new_level = level
        self.old_level = None
    
    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.level = self.new_level
        self.logger.current_level = self.logger.levels.get(self.new_level, 1)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.level = self.old_level
        self.logger.current_level = self.logger.levels.get(self.old_level, 1)

# Specific debug functions for your PHI-Seg model
def debug_phiseg_forward(model, x, mask=None, logger=None):
    """Debug PHI-Seg forward pass step by step"""
    if logger is None:
        logger = debug_logger
    
    logger.info("=== PHI-Seg Forward Debug ===")
    debug_tensor(x, "input_x", logger, detailed=True)
    if mask is not None:
        debug_tensor(mask, "input_mask", logger, detailed=True)
    
    # Debug prior net
    logger.info("--- Prior Net ---")
    if hasattr(model, 'prior_net'):
        # Add temporary hooks to prior net
        hooks = []
        for name, module in model.prior_net.named_modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d)):
                hook = module.register_forward_hook(debug_forward_hook(f"prior.{name}", logger))
                hooks.append(hook)
        
        try:
            # This will trigger the debug hooks
            with torch.no_grad():
                prior_result = model.prior_net(x, generation_mode=True)
                debug_tensor(prior_result[0], "prior_z", logger)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

def debug_channel_flow(model, x, logger=None):
    """Debug channel dimensions through the model"""
    if logger is None:
        logger = debug_logger
    
    logger.info("=== Channel Flow Debug ===")
    debug_tensor(x, "initial_input", logger)
    
    # Check encoder blocks
    if hasattr(model, 'prior_net') and hasattr(model.prior_net, 'encoder_blocks'):
        current = x
        for i, block in enumerate(model.prior_net.encoder_blocks):
            logger.info(f"--- Encoder Block {i} ---")
            debug_module(block, f"encoder_block_{i}", logger)
            
            try:
                with torch.no_grad():
                    current = block(current)
                    debug_tensor(current, f"encoder_out_{i}", logger)
            except Exception as e:
                logger.error(f"Error in encoder block {i}: {e}")
                break
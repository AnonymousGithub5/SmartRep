function FunctionDefinition_0 ( IERC20 _token ) external onlyOwner { uint256 balance = _token . balanceOf ( this ) ; _token . safeTransfer ( owner , balance ) ; }
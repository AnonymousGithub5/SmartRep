function FunctionDefinition_0 ( address account , uint256 amount ) internal { require ( account != address ( 0 ) , stringLiteral_0 ) ; totalSupply = totalSupply . add ( amount ) ; balanceOf [ account ] = balanceOf [ account ] . add ( amount ) ; emit Transfer ( address ( 0 ) , account , amount ) ; }
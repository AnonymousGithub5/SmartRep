function withdraw ( uint256 _amount ) public { PoolInfo storage pool = poolInfo [ 0 ] ; UserInfo storage user = userInfo [ msg . sender ] ; require ( user . amount >= _amount , stringLiteral_0 ) ; Identifier_0 ( 0 ) ; uint256 VariableDeclaration_0 = user . amount . mul ( pool . MemberAccess_0 ) . div ( 1e12 ) . sub ( user . rewardDebt ) ; if ( pending > 0 ) { Identifier_1 ( Identifier_2 ) . withdraw ( pending ) ; Identifier_3 ( address ( msg . sender ) , pending ) ; } if ( _amount > 0 ) { user . amount = user . amount . sub ( _amount ) ; pool . MemberAccess_1 . safeTransfer ( address ( msg . sender ) , _amount ) ; } user . rewardDebt = user . amount . mul ( pool . MemberAccess_2 ) . div ( 1e12 ) ; emit Identifier_4 ( msg . sender , _amount ) ; }
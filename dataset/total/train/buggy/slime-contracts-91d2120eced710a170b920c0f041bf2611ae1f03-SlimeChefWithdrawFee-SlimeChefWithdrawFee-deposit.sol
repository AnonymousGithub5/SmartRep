function deposit ( uint256 _amount ) external { PoolInfo storage pool = poolInfo [ 0 ] ; UserInfo storage user = userInfo [ msg . sender ] ; Identifier_0 ( 0 ) ; if ( user . amount > 0 ) { uint256 VariableDeclaration_0 = user . amount . mul ( pool . MemberAccess_0 ) . div ( 1e12 ) . sub ( user . rewardDebt ) ; user . rewardDebt = user . amount . mul ( pool . MemberAccess_1 ) . div ( 1e12 ) ; user . amount = user . amount . add ( _amount ) ; if ( pending > 0 ) { Identifier_1 . safeTransfer ( address ( msg . sender ) , pending ) ; } } if ( _amount > 0 ) { pool . MemberAccess_2 . safeTransferFrom ( address ( msg . sender ) , address ( this ) , _amount ) ; } emit Identifier_2 ( msg . sender , _amount ) ; }
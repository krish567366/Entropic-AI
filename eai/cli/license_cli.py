#!/usr/bin/env python3
"""
Entropic AI License Management CLI

Command-line tool for managing E-AI licenses.
"""

import click
import json
import os
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

from eai.licensing import (
    get_license_manager, 
    LicenseError,
    show_license_info,
    check_license_status
)


@click.group()
@click.version_option()
def cli():
    """Entropic AI License Management Tool"""
    pass


@cli.command()
def status():
    """Show license status information."""
    try:
        show_license_info()
    except Exception as e:
        click.echo(f"❌ Error checking license: {e}", err=True)


@cli.command()
@click.argument('license_file')
def activate(license_file):
    """Activate a license from file."""
    license_path = Path(license_file)
    
    if not license_path.exists():
        click.echo(f"❌ License file not found: {license_file}", err=True)
        return
    
    # Create license directory
    license_dir = Path.home() / '.eai'
    license_dir.mkdir(exist_ok=True)
    
    # Copy license file
    target_path = license_dir / 'license.json'
    
    try:
        with open(license_path, 'r') as src, open(target_path, 'w') as dst:
            license_data = json.load(src)
            json.dump(license_data, dst, indent=2)
        
        click.echo(f"✅ License activated successfully!")
        click.echo(f"📁 Installed to: {target_path}")
        
        # Show license info
        show_license_info()
        
    except Exception as e:
        click.echo(f"❌ Failed to activate license: {e}", err=True)


@cli.command()
def deactivate():
    """Deactivate current license."""
    license_path = Path.home() / '.eai' / 'license.json'
    
    if license_path.exists():
        license_path.unlink()
        click.echo("✅ License deactivated successfully")
    else:
        click.echo("ℹ️  No active license found")


@cli.command()
def validate():
    """Validate current license."""
    try:
        manager = get_license_manager()
        info = manager.get_license_info()
        
        if info["status"] == "valid":
            click.echo("✅ License is valid")
            click.echo(f"Type: {info['license_type']}")
            click.echo(f"Features: {', '.join(info['features'])}")
            click.echo(f"Expires: {info['expiry_date']}")
        else:
            click.echo(f"❌ License validation failed: {info['status']}")
            
    except LicenseError as e:
        click.echo(f"❌ License error: {e}", err=True)


@cli.command()
def features():
    """List available features in current license."""
    try:
        manager = get_license_manager()
        available_features = manager.get_available_features()
        
        if available_features:
            click.echo("🔧 Available Features:")
            for feature in available_features:
                click.echo(f"  • {feature}")
        else:
            click.echo("❌ No features available (invalid or expired license)")
            
    except LicenseError as e:
        click.echo(f"❌ License error: {e}", err=True)


@cli.command()
def purchase():
    """Get information about purchasing a license."""
    click.echo("\n💰 Entropic AI Licensing Information")
    click.echo("=" * 50)
    click.echo("\n📧 Contact Sales:")
    click.echo("   Email: bajpaikrishna715@gmail.com")
    click.echo("   Web: https://entropicai.com/licensing")
    click.echo("   Phone: +1-XXX-XXX-XXXX")
    click.echo("\n💼 License Types:")
    click.echo("   • Academic: For research and education")
    click.echo("     - Basic networks and optimization")
    click.echo("     - Visualization tools")
    click.echo("     - Educational tutorials")
    click.echo("\n   • Professional: For commercial development")
    click.echo("     - All academic features")
    click.echo("     - Advanced networks and algorithms")
    click.echo("     - Molecule and circuit evolution")
    click.echo("     - API access")
    click.echo("\n   • Enterprise: Full platform access")
    click.echo("     - All professional features")
    click.echo("     - Theory discovery capabilities")
    click.echo("     - Custom applications")
    click.echo("     - Distributed computing")
    click.echo("     - Priority support")
    click.echo("=" * 50)


@cli.command()
@click.option('--days', default=7, help='Check license expiry within N days')
def check_expiry(days):
    """Check if license expires within specified days."""
    try:
        manager = get_license_manager()
        info = manager.get_license_info()
        
        if info["status"] == "valid":
            days_remaining = info["days_remaining"]
            
            if days_remaining <= days:
                click.echo(f"⚠️  License expires in {days_remaining} days!")
                click.echo("Contact bajpaikrishna715@gmail.com for renewal")
            else:
                click.echo(f"✅ License valid for {days_remaining} days")
        else:
            click.echo(f"❌ License status: {info['status']}")
            
    except LicenseError as e:
        click.echo(f"❌ License error: {e}", err=True)


@cli.command()
def health():
    """Comprehensive license health check."""
    try:
        manager = get_license_manager()
        health = manager.check_license_health()
        
        if health["healthy"]:
            click.echo("✅ License is healthy")
        else:
            click.echo("❌ License has issues")
        
        click.echo(f"Status: {health['status']}")
        click.echo(f"Message: {health['message']}")
        
        if health["action_required"]:
            click.echo(f"Action Required: {health['action_required']}")
            
    except Exception as e:
        click.echo(f"❌ Health check failed: {e}", err=True)


@cli.command()
@click.option('--output', '-o', help='Output file for license request')
def request():
    """Generate a license request file."""
    import platform
    import uuid
    
    # Generate hardware fingerprint
    machine_id = str(uuid.getnode())
    system_info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'node': platform.node(),
        'machine_id': machine_id
    }
    
    # Create license request
    request_data = {
        'request_id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'system_info': system_info,
        'requested_features': ['core'],  # Default request
        'contact_info': {
            'name': click.prompt('Your name'),
            'email': click.prompt('Your email'),
            'organization': click.prompt('Organization (optional)', default=''),
        }
    }
    
    output_file = click.get_text_stream('stdout') if not click.get_current_context().params.get('output') else open(click.get_current_context().params['output'], 'w')
    
    json.dump(request_data, output_file, indent=2)
    
    if output_file != click.get_text_stream('stdout'):
        output_file.close()
        click.echo(f"✅ License request saved to: {click.get_current_context().params['output']}")
    
    click.echo("\n📧 Send this request to: bajpaikrishna715@gmail.com")


if __name__ == '__main__':
    cli()
